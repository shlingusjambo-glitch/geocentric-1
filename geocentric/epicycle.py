"""EPICYCLE: progressive depth, context folding, trimmed loss and compact state.

These experimental combinations build on prior training research. See EPICYCLE.md
for mechanisms, numerical tradeoffs, persistence guarantees and measured results.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class EpicycleConfig:
    enabled: bool = False

    # --- DEFERENT: elastic depth ---------------------------------------------
    deferent: bool = True
    deferent_start: float = 0.5   # fraction of layers live at step 0
    deferent_full_by: float = 0.6  # fraction of training by which all layers are live

    # --- HORIZON: elastic context --------------------------------------------
    horizon: bool = True
    horizon_start: int = 256      # context length at step 0
    horizon_full_by: float = 0.5  # fraction of training by which context is full
    horizon_mode: str = "fold"   # fold preserves every target; crop is legacy

    # --- EQUANT: trimmed-band token selection --------------------------------
    equant: bool = True
    equant_keep: float = 0.65     # fraction of positions that contribute gradient
    equant_trim: float = 0.02     # fraction discarded off the *top* as corrupt/outlier
    equant_after: float = 0.15    # fraction of training to wait before selecting

    # --- ARMILLARY: rotating optimizer state ---------------------------------
    armillary: bool = False       # off by default: it trades a little quality for memory
    armillary_rings: int = 4
    armillary_dwell: int = 200    # steps a ring stays hot before the next takes over
    armillary_factored: bool = False  # experimental row/column second moments
    armillary_partitioned: bool = False  # opt-in: split tensors across momentum rings
    equant_sparse_replay: bool = False  # eager selected-row vocabulary backward

    def __post_init__(self):
        if not 0 < self.deferent_start <= 1:
            raise ValueError("deferent_start must be in (0, 1]")
        if not 0 < self.deferent_full_by <= 1 or not 0 < self.horizon_full_by <= 1:
            raise ValueError("full_by fractions must be in (0, 1]")
        if self.horizon_start < 1 or self.horizon_mode not in {"fold", "crop"}:
            raise ValueError("horizon_start must be positive and horizon_mode fold or crop")
        if not 0 < self.equant_keep <= 1 or not 0 <= self.equant_trim < 1:
            raise ValueError("Invalid EQUANT keep/trim fractions")
        if self.equant_keep + self.equant_trim > 1 or not 0 <= self.equant_after <= 1:
            raise ValueError("EQUANT keep + trim must be <= 1 and after in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EpicycleConfig":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def preset(cls, name: str) -> "EpicycleConfig":
        """Named settings, because four independent gears is three too many to tune."""
        name = (name or "off").strip().lower()
        if name in {"off", "none"}:
            return cls(enabled=False)
        if name == "speed":
            # Reduce early attention/depth work; conditioning and layer update counts change.
            return cls(enabled=True, deferent=True, horizon=True, equant=False, armillary=False)
        if name == "quality":
            return cls(enabled=True, deferent=False, horizon=True, equant=True, armillary=False)
        if name == "selective":
            return cls(enabled=True, deferent=False, horizon=True, equant=True,
                       armillary=False, equant_sparse_replay=True)
        if name == "memory":
            # Buy parameters with optimizer state.
            return cls(enabled=True, deferent=True, horizon=True, equant=False, armillary=True)
        if name == "full":
            return cls(enabled=True, deferent=True, horizon=True, equant=True, armillary=True)
        if name == "capacity":
            return cls(enabled=True, equant=False, armillary=True, armillary_factored=True)
        if name == "balanced":
            return cls(enabled=True, equant=False, armillary=True, armillary_factored=True,
                       armillary_partitioned=True)
        raise ValueError(f"Unknown epicycle preset {name!r}. Choose off, speed, quality, memory, full, capacity, balanced, selective.")


class EpicycleScheduler:
    """Drives DEFERENT, HORIZON and EQUANT from the optimizer step count.

    Holds no state the model needs at inference: switch it off mid-run and training
    continues as a plain run of the same model.
    """

    def __init__(
        self,
        config: EpicycleConfig,
        model: nn.Module,
        total_steps: int,
        block_size: int,
        start_step: int = 0,
    ) -> None:
        self.config = config
        self.model = model
        self.total_steps = max(1, total_steps)
        self.block_size = block_size
        self.n_layer = len(model.blocks)
        self.events: List[str] = []
        # Never reinitialize layers that have already learned on a resumed run.
        # The initial active stack keeps its scaled random initialization. Sleeping
        # blocks are identities so full-depth validation also measures the live model.
        self._grown_to = self.active_layers(start_step)
        saved = getattr(model, "_epicycle_state", None)
        if saved:
            self._grown_to = max(self._grown_to, saved["grown_to"])
        if start_step == 0 and not saved and config.enabled and config.deferent:
            self._zero_blocks(self._grown_to, self.n_layer)
        self._last_ctx = block_size
        self.apply(start_step)

    # -- DEFERENT -----------------------------------------------------------
    def active_layers(self, step: int) -> int:
        c = self.config
        if not (c.enabled and c.deferent):
            return self.n_layer
        progress = step / (self.total_steps * max(1e-6, c.deferent_full_by))
        frac = c.deferent_start + (1.0 - c.deferent_start) * min(1.0, progress)
        return max(1, min(self.n_layer, math.ceil(self.n_layer * frac)))

    def _grow_to(self, depth: int) -> None:
        """Activate blocks up to `depth`, each as an exact identity map.

        Zeroing the two residual output projections makes a freshly activated block
        contribute nothing on the step it appears, so growth is function-preserving
        and the loss curve has no step in it. The block is not frozen: its `down` and
        `proj` weights get a real gradient immediately (the loss depends on them
        directly even at zero), so it leaves identity on the very next update.
        """
        self._zero_blocks(self._grown_to, depth)
        self._grown_to = max(self._grown_to, depth)

    def _zero_blocks(self, start: int, end: int) -> None:
        for i in range(start, end):
            block = self.model.blocks[i]
            with torch.no_grad():
                block.attn.proj.weight.zero_()
                block.mlp.down.weight.zero_()
                if block.attn.proj.bias is not None:
                    block.attn.proj.bias.zero_()
                if block.mlp.down.bias is not None:
                    block.mlp.down.bias.zero_()

    # -- HORIZON ------------------------------------------------------------
    def context_length(self, step: int) -> int:
        c = self.config
        if not (c.enabled and c.horizon) or c.horizon_start >= self.block_size:
            return self.block_size
        progress = step / (self.total_steps * max(1e-6, c.horizon_full_by))
        if progress >= 1:
            return self.block_size  # including non-power-of-two final contexts
        span = self.block_size - c.horizon_start
        raw = c.horizon_start + span * min(1.0, progress)
        # Powers of two only: every change is a fresh kernel autotune and, under
        # torch.compile, a recompilation. Four steps beats a thousand.
        ctx = 1 << int(math.floor(math.log2(max(2, raw))))
        return max(c.horizon_start, min(self.block_size, ctx))

    # -- EQUANT -------------------------------------------------------------
    def selecting(self, step: int) -> bool:
        c = self.config
        return bool(c.enabled and c.equant and step >= self.total_steps * c.equant_after)

    # -- driver -------------------------------------------------------------
    def apply(self, step: int) -> Tuple[int, int]:
        """Set the model's depth for this step; return (active_layers, context)."""
        depth = self.active_layers(step)
        if self.config.enabled and self.config.deferent:
            if depth > self._grown_to:
                self._grow_to(depth)
                if step > 0:
                    self.events.append(f"step {step}: depth {depth}/{self.n_layer}")
            self.model.active_layers = depth if depth < self.n_layer else None
        ctx = self.context_length(step)
        if ctx != self._last_ctx and step > 0:
            self.events.append(f"step {step}: context {ctx}")
        self._last_ctx = ctx
        self.model._epicycle_state = {"grown_to": self._grown_to,
                                      "config": self.config.to_dict()}
        return depth, ctx

    def prepare_batch(self, input_ids: torch.Tensor, labels: torch.Tensor, ctx: int):
        """HORIZON folding: pay short-context attention without discarding targets.

        Every original (input, next-token target) pair occurs exactly once. Each
        segment gets a fresh attention context; only a final partial segment is
        padded, with ignored targets. The batch's total supervised tokens is fixed.
        """
        if self.config.horizon_mode == "crop":
            return self.crop(input_ids, labels, ctx)
        if ctx < 1:
            raise ValueError("ctx must be positive")
        if ctx >= input_ids.size(1):
            return input_ids, labels
        import torch.nn.functional as F

        pad = (-input_ids.size(1)) % ctx
        if pad:
            input_ids = F.pad(input_ids, (0, pad), value=0)
            labels = F.pad(labels, (0, pad), value=-100)
        return input_ids.reshape(-1, ctx), labels.reshape(-1, ctx)

    def crop(self, input_ids: torch.Tensor, labels: torch.Tensor, ctx: int):
        if ctx >= input_ids.size(1):
            return input_ids, labels
        # Slicing from the front is valid because RoPE is relative: positions
        # 0..ctx-1 of a window are a legitimate shorter window on their own.
        return input_ids[:, :ctx].contiguous(), labels[:, :ctx].contiguous()

    def status(self, step: int) -> str:
        depth, ctx = self.active_layers(step), self.context_length(step)
        bits = []
        if self.config.deferent and depth < self.n_layer:
            bits.append(f"L{depth}/{self.n_layer}")
        if self.config.horizon and ctx < self.block_size:
            bits.append(f"ctx{ctx}")
        if self.selecting(step):
            bits.append(f"eq{self.config.equant_keep:.0%}")
        return " ".join(bits)


def equant_loss(
    per_token: torch.Tensor,
    labels: torch.Tensor,
    keep: float = 0.65,
    trim: float = 0.02,
) -> torch.Tensor:
    """Reduce a per-token loss vector to a scalar over a trimmed high-loss band.

    Once a model predicts a token confidently, the gradient that token contributes is
    near zero but the *variance* it contributes is not: it dilutes the batch. Keeping
    only the hardest tokens concentrates the update where the model is still wrong.

    The trim is the part that matters and the part naive hard-example mining misses.
    The very highest-loss tokens in a web corpus are not hard, they are broken —
    mojibake, truncated unicode, base64 in the middle of an article, a table's worth
    of numbers. Selecting purely by top-k steers the model straight into them. So the
    top `trim` fraction is discarded and the band below it is kept.
    """
    valid = labels.reshape(-1) != -100
    if not 0 < keep <= 1 or not 0 <= trim < 1 or keep + trim > 1:
        raise ValueError("Require 0 < keep <= 1, 0 <= trim < 1, keep + trim <= 1")
    losses = per_token[valid]
    n = losses.numel()
    if n < 16:
        return losses.mean() if n else per_token.sum() * 0.0

    n_trim = int(n * trim)
    n_keep = max(1, int(n * keep))
    if n_trim + n_keep > n:
        n_keep = n - n_trim
    # One sort beats two topk calls and gives the band directly.
    ordered, _ = torch.sort(losses, descending=True)
    return ordered[n_trim : n_trim + n_keep].mean()


class RingAdamW(Optimizer):
    """ARMILLARY — AdamW whose momentum lives on a rotating subset of the weights.

    Standard FP32 AdamW costs 16 bytes per parameter including gradients: 4 each
    for the weight, gradient, first moment and second moment. On a memory-bound card that state, not
    the arithmetic, is what caps model size.

    Two changes:

    - The second moment is stored bf16 (2 bytes). It is a slowly-moving positive
      scale that gets square-rooted before use, so it tolerates 8 mantissa bits;
      this is the same bet Adafactor and 8-bit Adam make but rounding changes the optimizer numerically.
    - The first moment exists for one *ring* of parameters at a time. Parameters are
      dealt into `rings` groups; ring `r` holds momentum for `dwell` consecutive
      steps and then hands it to the next ring, freeing its buffer. Cold parameters
      still update every step — they take a momentum-free Adam step, which is plain
      RMSProp — so nothing goes stale, it simply loses smoothing while cold.

    Counted the way a memory budget actually fills up — weight, gradient, and both
    moments — fp32 AdamW costs 16 bytes per parameter and this costs 4 + 4 + 2 +
    4/rings. At rings=4 that is 11 against 16, so the same budget holds about 1.45x
    the parameters. (Optimizer state alone goes from 8 bytes to 3, but the gradient
    buffer is just as real, so quoting the state in isolation overstates the win.)

    The dwell has to be long relative to momentum's own horizon (1/(1-beta1) ~ 10
    steps) or momentum never accumulates before it is freed; 200 is comfortable.

    rings=1 disables the rotation and leaves you plain AdamW with a bf16 second
    moment — 14 bytes per parameter including gradients. It is still an approximation.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        rings: int = 4,
        dwell: int = 200,
        factored: bool = False,
        partitioned: bool = False,
    ) -> None:
        if rings < 1:
            raise ValueError(f"rings must be at least 1, got {rings}")
        if dwell < 1:
            raise ValueError(f"dwell must be at least 1, got {dwell}")
        if lr < 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid optimizer learning rate, epsilon, decay or betas")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.rings = rings
        self.dwell = dwell
        self.factored = factored
        self.partitioned = partitioned
        self._global_step = 0
        self._assign_rings()

    def _assign_rings(self) -> None:
        """Deal parameters into rings, largest first, always into the lightest ring.

        Round-robin would put every embedding matrix in one ring and every norm gain
        in another, so one ring would carry most of the momentum memory and the peak
        would be no better than plain AdamW. Greedy balancing cannot split a dominant
        tensor: the largest tensor is a lower bound on the largest ring.
        """
        tensors = [p for group in self.param_groups for p in group["params"]]
        load = [0] * self.rings
        for p in sorted(tensors, key=lambda t: -t.numel()):
            r = min(range(self.rings), key=lambda i: load[i])
            self.state[p]["ring"] = r
            load[r] += p.numel()
        self.ring_load = load
        if self.partitioned:
            self.ring_load = [sum(p.numel() * (r + 1) // self.rings - p.numel() * r // self.rings
                                  for p in tensors) for r in range(self.rings)]

    @property
    def hot_ring(self) -> int:
        return (self._global_step // self.dwell) % self.rings

    def state_bytes(self) -> int:
        total = 0
        for state in self.state.values():
            for key in ("v", "m", "v_row", "v_col"):
                tensor = state.get(key)
                if torch.is_tensor(tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    def state_dict(self):
        result = super().state_dict()
        result["armillary"] = dict(global_step=self._global_step, rings=self.rings,
                                   dwell=self.dwell, factored=self.factored, partitioned=self.partitioned)
        return result

    def load_state_dict(self, state_dict):
        meta = state_dict.get("armillary", {})
        if meta and (meta["rings"] != self.rings or meta["dwell"] != self.dwell
                     or meta.get("factored", False) != self.factored
                     or meta.get("partitioned", False) != self.partitioned):
            raise ValueError("ARMILLARY configuration differs from the saved optimizer")
        # Optimizer.load_state_dict casts floating state to parameter dtype. Restore
        # deliberate storage dtypes afterwards, or resume silently doubles v memory.
        super().load_state_dict(state_dict)
        self._global_step = meta.get("global_step", max(
            (s.get("t", 0) for s in self.state.values()), default=0))
        for state in self.state.values():
            if "v" in state:
                state["v"] = state["v"].to(torch.bfloat16)
            for key in ("m", "v_row", "v_col"):
                if key in state:
                    state[key] = state[key].float()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        hot = self.hot_ring
        # Free the entire old ring BEFORE allocating any of the new one, including
        # parameters with no gradient. Interleaved freeing can transiently hold two.
        for state in self.state.values():
            resident = state.get("m_ring") if self.partitioned else state.get("ring")
            if resident != hot:
                state.pop("m", None)
                state["mt"] = 0
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    # A DEFERENT-inactive block lands here: no gradient, no update,
                    # and — crucially — no decoupled weight decay either.
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RingAdamW does not support sparse gradients.")
                state = self.state[p]
                factor = self.factored and p.ndim >= 2 and min(p.shape) > 1
                if "t" not in state:
                    if factor:
                        state["v_row"] = torch.zeros(p.numel() // p.shape[-1], device=p.device)
                        state["v_col"] = torch.zeros(p.shape[-1], device=p.device)
                    else:
                        state["v"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["t"] = 0
                    state.setdefault("ring", 0)
                state["t"] += 1

                grad = grad.float()
                if factor:
                    squared = grad.reshape(-1, p.shape[-1]).square()
                    row, col = state["v_row"], state["v_col"]
                    row.lerp_(squared.mean(dim=1), 1 - beta2)
                    col.lerp_(squared.mean(dim=0), 1 - beta2)
                    del squared
                    v = ((row / row.mean().clamp_min(eps * eps))[:, None]
                         * col[None, :]).reshape_as(p)
                else:
                    v = state["v"].float()
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    state["v"].copy_(v)
                v.div_(1 - beta2 ** state["t"])
                denom = v.sqrt_().add_(eps)

                if self.partitioned:
                    start = p.numel() * hot // self.rings
                    end = p.numel() * (hot + 1) // self.rings
                    update = (grad / denom).contiguous()
                    if end > start:
                        if "m" not in state:
                            state["m"] = torch.zeros(end - start, device=p.device, dtype=torch.float32)
                            state["mt"] = 0
                            state["m_ring"] = hot
                        m = state["m"]
                        m.mul_(beta1).add_(grad.reshape(-1)[start:end], alpha=1 - beta1)
                        state["mt"] += 1
                        update.reshape(-1)[start:end].copy_(
                            (m / (1 - beta1 ** state["mt"])).div_(denom.reshape(-1)[start:end]))
                elif state["ring"] == hot:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["mt"] = 0
                    m = state["m"]
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["mt"] += 1
                    update = (m / (1 - beta1 ** state["mt"])).div_(denom)
                else:
                    if "m" in state:
                        # Hand the buffer back. The caching allocator recycles it into
                        # the ring that just went hot, so the peak stays at one ring.
                        del state["m"]
                        state["mt"] = 0
                    update = grad / denom

                if wd:
                    p.add_(p, alpha=-lr * wd)
                if factor:
                    # Factoring can underestimate individual coordinates. Bound the
                    # tensor update RMS as in Adafactor; this is deliberately an
                    # experimental optimizer, not numerically equivalent AdamW.
                    update.div_(update.square().mean().sqrt().clamp_min(1.0))
                p.add_(update, alpha=-lr)

        self._global_step += 1
        return loss


def build_epicycle_optimizer(
    model: nn.Module,
    config: EpicycleConfig,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    quiet: bool = False,
):
    """RingAdamW with the same decay/no-decay split the standard path uses."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = RingAdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=betas, rings=config.armillary_rings, dwell=config.armillary_dwell,
        factored=config.armillary_factored,
        partitioned=config.armillary_partitioned,
    )
    if not quiet:
        n = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
        # weight + gradient + bf16 second moment + fp32 first moment on one ring,
        # against weight + gradient + two fp32 moments.
        second_bytes = sum(
            4 * (p.numel() // p.shape[-1] + p.shape[-1])
            if config.armillary_factored and p.ndim >= 2 and min(p.shape) > 1
            else 2 * p.numel() for p in decay + no_decay
        )
        per_param = 8 + (second_bytes + 4 * max(optimizer.ring_load)) / n
        baseline = 16.0
        print(
            f"ARMILLARY: {config.armillary_rings} rings, dwell {config.armillary_dwell} steps | "
            f"~{per_param:.1f} bytes/param against {baseline:.0f} for fp32 AdamW "
            f"({baseline / per_param:.2f}x the parameters in the same budget, "
            f"{n * (baseline - per_param) / 1024**2:.0f} MB saved on this model)"
        )
    return optimizer
