"""Watch the loss curve and act when it does something that is not noise.

A from-scratch run does not descend smoothly. It descends, wobbles, occasionally
jumps by a whole nat when a batch of mojibake or a truncated table goes through, and
very occasionally diverges and never recovers. Left alone, one bad batch at hour
thirty can undo hour twenty-nine, and nothing tells you until you look.

The temptation is to roll back whenever the loss goes up. That is a mistake and it is
worth being explicit about why. The reported loss is an average over one accumulation
window, so it is a noisy estimate of a quantity that is itself noisy; a cosine
schedule raises it legitimately at transitions; and EQUANT changes what is being
averaged. A guard that reacts to every uptick reverts real progress on random noise
and converts training into a walk that never advances.

So this guard reacts to *abnormality*, not to direction:

  spike        one window far above the running distribution -> drop that update
  persistence  several in a row -> roll back to a known-good checkpoint, re-warm the LR
  divergence   sustained rise over hundreds of steps -> stop, with a reason
  plateau      no improvement over a long window -> reported, never acted on

Everything it does is recorded in training_metrics.json, so `geocentric bench` can
tell you afterwards that the run had four spikes and a rollback at step 8,200 rather
than leaving you to guess why the scores came out low.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Actions the trainer must understand. Anything else is a bug in this file.
OK = "ok"
SKIP = "skip"          # drop this update; the gradient is suspect
ROLLBACK = "rollback"  # restore the last known-good state and re-warm
STOP = "stop"          # the run is diverging; further steps are wasted


@dataclass
class LossGuardConfig:
    enabled: bool = True

    # A spike is judged against the recent distribution, not against a fixed number:
    # a loss of 4.0 is normal at step 100 and a catastrophe at step 100,000.
    spike_sigma: float = 4.0
    min_spike_delta: float = 0.15   # never call a tiny wobble a spike, however tight the variance
    warmup_steps: int = 50          # no verdicts before there is a distribution to compare to

    skip_bad_updates: bool = True
    max_consecutive_spikes: int = 4  # this many in a row is not bad luck
    rollback: bool = True
    max_rollbacks: int = 3
    rewarm_steps: int = 100          # LR ramp back after a rollback

    divergence_window: int = 300
    divergence_ratio: float = 1.30
    stop_on_divergence: bool = False  # off by default: an unattended run should not exit

    plateau_window: int = 2000
    plateau_delta: float = 0.005

    # EMA horizon for the running mean and variance, in optimizer steps.
    horizon: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LossGuardConfig":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Verdict:
    action: str = OK
    reason: str = ""
    lr_scale: float = 1.0

    @property
    def normal(self) -> bool:
        return self.action == OK and self.lr_scale == 1.0


class LossGuard:
    """Running statistics over the loss, plus the decisions that follow from them."""

    def __init__(self, config: Optional[LossGuardConfig] = None, start_step: int = 0) -> None:
        self.config = config or LossGuardConfig()
        self.mean: Optional[float] = None
        self.var: float = 0.0
        self.best: float = float("inf")
        self.best_step: int = start_step
        self.seen: int = 0
        self.start_step = start_step

        self.consecutive_spikes = 0
        self.spikes: List[Dict[str, Any]] = []
        self.rollbacks: List[Dict[str, Any]] = []
        self.skipped_updates = 0
        self.plateau_since: Optional[int] = None
        self.exhausted_at: Optional[int] = None
        self._rewarm_until: int = -1
        self._window_best: float = float("inf")
        self._window_start: int = start_step

    # -- statistics ---------------------------------------------------------
    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.var))

    def _update_stats(self, loss: float) -> None:
        alpha = 2.0 / (self.config.horizon + 1)
        if self.mean is None:
            self.mean, self.var = loss, 0.0
        else:
            delta = loss - self.mean
            self.mean += alpha * delta
            self.var = (1 - alpha) * (self.var + alpha * delta * delta)
        self.seen += 1

    def threshold(self) -> Optional[float]:
        if self.mean is None or self.seen < self.config.warmup_steps:
            return None
        return self.mean + max(self.config.spike_sigma * self.std, self.config.min_spike_delta)

    # -- the decision -------------------------------------------------------
    def observe(self, step: int, loss: float) -> Verdict:
        """Judge one optimizer step's loss. Call this *before* applying the update."""
        config = self.config
        if not config.enabled:
            return Verdict()

        # Once the rollback budget is gone the guard has demonstrably failed to fix
        # whatever is wrong, and continuing to drop every update would stall the run
        # entirely — a worse outcome than letting the gradients through. So it stands
        # down, lets the statistics absorb the new regime, and says so once.
        exhausted = config.rollback and len(self.rollbacks) >= config.max_rollbacks
        if exhausted and self.exhausted_at is None:
            self.exhausted_at = step

        if not math.isfinite(loss):
            # Never let a NaN through, exhausted or not: it would poison every weight
            # in one step, which is not a recoverable state at any budget.
            self.consecutive_spikes += 1
            self.skipped_updates += 1
            self.spikes.append({"step": step, "loss": None, "kind": "non-finite"})
            if (self.consecutive_spikes >= config.max_consecutive_spikes
                    and config.rollback and not exhausted):
                return self._request_rollback(step, "loss became non-finite repeatedly")
            return Verdict(SKIP, "non-finite loss", self._lr_scale(step))

        limit = self.threshold()
        is_spike = limit is not None and loss > limit

        if is_spike and not exhausted:
            self.consecutive_spikes += 1
            self.spikes.append({
                "step": step, "loss": round(loss, 4),
                "threshold": round(limit, 4), "mean": round(self.mean or 0.0, 4),
                "sigma_above": round((loss - (self.mean or 0.0)) / max(1e-6, self.std), 2),
            })
            # A spike must not enter the statistics, or a big enough one raises the
            # threshold and the next spike looks normal.
            if (self.consecutive_spikes >= config.max_consecutive_spikes
                    and config.rollback and len(self.rollbacks) < config.max_rollbacks):
                return self._request_rollback(
                    step,
                    f"{self.consecutive_spikes} consecutive loss spikes "
                    f"(latest {loss:.4f} against a threshold of {limit:.4f})",
                )
            if config.skip_bad_updates:
                self.skipped_updates += 1
                return Verdict(
                    SKIP,
                    f"loss {loss:.4f} is {(loss - self.mean) / max(1e-6, self.std):.1f} sigma "
                    f"above the running mean {self.mean:.4f}; dropping this update",
                    self._lr_scale(step),
                )
            return Verdict(OK, "spike recorded but not acted on", self._lr_scale(step))

        if is_spike:
            # Exhausted: recorded, but allowed through so the run can move on.
            self.spikes.append({"step": step, "loss": round(loss, 4), "kind": "unguarded"})

        # Normal step.
        self.consecutive_spikes = 0
        self._update_stats(loss)
        if loss < self.best:
            self.best, self.best_step = loss, step

        # --- plateau (reported, never acted on) ---
        if loss < self._window_best - config.plateau_delta:
            self._window_best, self._window_start = loss, step
            self.plateau_since = None
        elif step - self._window_start >= config.plateau_window:
            self.plateau_since = self._window_start

        # --- divergence ---
        if (not exhausted
                and self.seen > config.divergence_window
                and self.best < float("inf")
                and self.mean > self.best * config.divergence_ratio
                and step - self.best_step > config.divergence_window):
            reason = (
                f"running mean {self.mean:.4f} has stayed {self.mean / self.best:.2f}x above the "
                f"best loss {self.best:.4f} (step {self.best_step:,}) for {step - self.best_step:,} steps"
            )
            if config.stop_on_divergence:
                return Verdict(STOP, reason, self._lr_scale(step))
            if config.rollback and len(self.rollbacks) < config.max_rollbacks:
                return self._request_rollback(step, reason)

        return Verdict(OK, "", self._lr_scale(step))

    def _request_rollback(self, step: int, reason: str) -> Verdict:
        self.rollbacks.append({"step": step, "reason": reason})
        self.consecutive_spikes = 0
        self._rewarm_until = step + self.config.rewarm_steps
        # Widen the distribution so the steps right after a rollback are not
        # immediately re-flagged by a variance estimate the spike itself narrowed.
        self.var = max(self.var, (self.config.min_spike_delta * 2) ** 2)
        return Verdict(ROLLBACK, reason, self._lr_scale(step))

    def _lr_scale(self, step: int) -> float:
        """Linear re-warm after a rollback.

        Dropping straight back to the full learning rate at the step that just blew up
        walks into the same batch region at the same rate. Ramping from a tenth over a
        hundred steps is what production runs do and it is the cheapest part of this.
        """
        if step >= self._rewarm_until:
            return 1.0
        # A rollback can restore weights to a step well before the one that
        # triggered it, which would otherwise send `remaining` above
        # rewarm_steps and `progress` negative -- a negative learning rate
        # that climbs the loss instead of descending it. Clamp to [0, 1].
        remaining = self._rewarm_until - step
        progress = 1.0 - remaining / max(1, self.config.rewarm_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * progress

    # -- reporting ----------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "best_loss": None if self.best == float("inf") else round(self.best, 4),
            "best_step": self.best_step,
            "running_mean": None if self.mean is None else round(self.mean, 4),
            "running_std": round(self.std, 4),
            "spikes": len(self.spikes),
            "skipped_updates": self.skipped_updates,
            "rollbacks": len(self.rollbacks),
            "rollback_steps": [r["step"] for r in self.rollbacks],
            "plateau_since_step": self.plateau_since,
            "exhausted_at_step": self.exhausted_at,
            "recent_spikes": self.spikes[-8:],
            "config": self.config.to_dict(),
        }

    def status(self) -> str:
        bits = []
        if self.spikes:
            bits.append(f"{len(self.spikes)} spike{'s' if len(self.spikes) != 1 else ''}")
        if self.rollbacks:
            bits.append(f"{len(self.rollbacks)} rollback{'s' if len(self.rollbacks) != 1 else ''}")
        if self.plateau_since is not None:
            bits.append(f"plateau since {self.plateau_since:,}")
        if self.exhausted_at is not None:
            bits.append("guard stood down")
        return " ".join(bits)


# ---------------------------------------------------------------------------
# Rollback support
# ---------------------------------------------------------------------------

class WeightSnapshot:
    """A CPU copy of the last known-good weights, for cheap rollback.

    Rolling back to the last on-disk checkpoint means losing up to `save_every`
    steps — a thousand by default, which on a real run is hours. A CPU snapshot costs
    4 bytes per parameter of host RAM (a gigabyte for a 250M model) and turns that
    into a couple of hundred steps.

    Weights only, deliberately. Adam's moments are poisoned by the same spike, but
    they decay with a horizon of roughly 1/(1-beta) steps and the learning-rate
    re-warm after a rollback covers exactly that window. Snapshotting them too would
    triple the cost for a correction that fixes itself.
    """

    def __init__(self, every: int = 200, enabled: bool = True) -> None:
        self.every = max(1, every)
        self.enabled = enabled
        self.step: Optional[int] = None
        self._state: Optional[Dict[str, Any]] = None

    @property
    def bytes(self) -> int:
        if not self._state:
            return 0
        return sum(t.numel() * t.element_size() for t in self._state.values() if hasattr(t, "numel"))

    def maybe_take(self, model, step: int, healthy: bool) -> bool:
        if not self.enabled or not healthy:
            return False
        if self.step is not None and step - self.step < self.every:
            return False
        self._state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        import copy
        self._epicycle_state = copy.deepcopy(getattr(model, "_epicycle_state", None))
        self.step = step
        return True

    def restore(self, model) -> Optional[int]:
        if not self._state:
            return None
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in self._state.items()})
        import copy
        model._epicycle_state = copy.deepcopy(self._epicycle_state)
        return self.step


def checkpoint_ledger_path(output_dir):
    from pathlib import Path

    return Path(output_dir) / "checkpoints.json"


def record_checkpoint(output_dir, name: str, step: int, loss=None, eval_loss=None) -> None:
    """Keep a small sidecar of what each checkpoint scored.

    Written so `--resume_from best` can compare candidates without loading a
    multi-gigabyte weights file twice at startup. A run from before this existed has
    no ledger, which the resume logic treats as "fall back to the latest checkpoint"
    rather than as an error.
    """
    import json

    path = checkpoint_ledger_path(output_dir)
    try:
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        ledger = {}
    entry = {"step": int(step)}
    if loss is not None and math.isfinite(float(loss)):
        entry["loss"] = round(float(loss), 6)
    if eval_loss is not None and math.isfinite(float(eval_loss)):
        entry["eval_loss"] = round(float(eval_loss), 6)
    ledger[name] = entry
    try:
        path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except OSError:
        pass


def choose_resume_checkpoint(output_dir, last_name: str, best_name: str, mode: str = "auto"):
    """Pick which checkpoint a resumed run should continue from.

    Returns (name, explanation). "auto" prefers the best checkpoint only when the
    ledger actually shows it is better — on held-out loss where both have one, on
    training loss otherwise. With no ledger, or nothing to compare, it takes the
    latest, which is what every earlier version of this code did.
    """
    import json
    from pathlib import Path

    out = Path(output_dir)
    has_last = (out / last_name).exists()
    has_best = (out / best_name).exists()
    mode = (mode or "auto").lower()

    if mode == "last" or not has_best:
        return (last_name if has_last else None), "latest checkpoint"
    if mode == "best" or not has_last:
        return best_name, "best checkpoint (explicitly requested)" if mode == "best" else "best checkpoint"

    ledger_path = checkpoint_ledger_path(out)
    if not ledger_path.exists():
        return last_name, ("latest checkpoint — no checkpoints.json, so this run predates "
                           "loss-aware resume and there is nothing to compare")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return last_name, "latest checkpoint — checkpoints.json unreadable"

    last, best = ledger.get(last_name, {}), ledger.get(best_name, {})
    for key, label in (("eval_loss", "held-out loss"), ("loss", "training loss")):
        if key in last and key in best:
            if best[key] < last[key]:
                return best_name, (
                    f"best checkpoint — {label} {best[key]:.4f} at step {best.get('step', 0):,} "
                    f"beats {last[key]:.4f} at step {last.get('step', 0):,}"
                )
            return last_name, (
                f"latest checkpoint — its {label} {last[key]:.4f} is already at or below the "
                f"best checkpoint's {best[key]:.4f}"
            )
    return last_name, "latest checkpoint — the ledger has no comparable loss for both"
