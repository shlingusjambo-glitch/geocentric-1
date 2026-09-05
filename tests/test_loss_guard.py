from __future__ import annotations

import random

import torch

from geocentric.loss_guard import (
    OK,
    ROLLBACK,
    SKIP,
    STOP,
    LossGuard,
    LossGuardConfig,
    WeightSnapshot,
    choose_resume_checkpoint,
    record_checkpoint,
)
from geocentric.model import GPTConfig, GeocentricGPT


def descending(guard: LossGuard, steps: int = 300, start: float = 6.0, noise: float = 0.05):
    """A realistic noisy descent — the curve the guard must never interrupt."""
    random.seed(11)
    verdicts = []
    for step in range(steps):
        loss = start * (0.995 ** step) + random.gauss(0, noise)
        verdicts.append(guard.observe(step, loss))
    return verdicts


# --- the thing that matters most: no false positives -----------------------

def test_a_normal_noisy_descent_is_never_interrupted():
    guard = LossGuard()
    assert all(v.action == OK for v in descending(guard))
    assert guard.spikes == [] and guard.rollbacks == []


def test_ordinary_upticks_are_not_spikes():
    """Loss going up is not, by itself, a problem. Rolling back on it would be."""
    guard = LossGuard()
    descending(guard, steps=200)
    baseline = guard.mean
    # A 5% rise, sustained — annoying to watch, entirely normal.
    for step in range(200, 260):
        assert guard.observe(step, baseline * 1.05).action == OK


def test_the_guard_stays_quiet_during_warmup():
    guard = LossGuard(LossGuardConfig(warmup_steps=50))
    for step in range(40):
        assert guard.observe(step, 10.0 if step % 7 else 2.0).action == OK
    assert guard.threshold() is None


# --- and it does fire on real trouble --------------------------------------

def test_a_single_spike_is_dropped_not_rolled_back():
    guard = LossGuard()
    descending(guard)
    verdict = guard.observe(300, guard.mean + 40 * max(guard.std, 0.1))
    assert verdict.action == SKIP
    assert guard.rollbacks == []
    assert guard.skipped_updates == 1


def test_consecutive_spikes_trigger_a_rollback_and_a_rewarm():
    config = LossGuardConfig(max_consecutive_spikes=3)
    guard = LossGuard(config)
    descending(guard)
    big = guard.mean + 40 * max(guard.std, 0.1)
    actions = [guard.observe(300 + i, big).action for i in range(3)]
    assert actions == [SKIP, SKIP, ROLLBACK]
    assert len(guard.rollbacks) == 1
    # The learning rate comes back gradually, not instantly.
    assert guard._lr_scale(303) < 0.2
    assert guard._lr_scale(303 + config.rewarm_steps) == 1.0


def test_a_spike_does_not_poison_the_running_statistics():
    """A big enough spike, folded into the mean, would hide the next one."""
    guard = LossGuard()
    descending(guard)
    mean_before, threshold_before = guard.mean, guard.threshold()
    guard.observe(300, 500.0)
    assert guard.mean == mean_before
    assert guard.threshold() == threshold_before


def test_non_finite_loss_is_always_dropped():
    guard = LossGuard()
    descending(guard)
    assert guard.observe(300, float("nan")).action == SKIP
    assert guard.observe(301, float("inf")).action == SKIP


def test_rollbacks_are_capped():
    config = LossGuardConfig(max_consecutive_spikes=2, max_rollbacks=2, rewarm_steps=1)
    guard = LossGuard(config)
    descending(guard)
    big = guard.mean + 50 * max(guard.std, 0.1)
    rollbacks = sum(1 for i in range(40) if guard.observe(300 + i, big).action == ROLLBACK)
    assert rollbacks == 2, "a run that cannot be saved must stop trying, not loop forever"


def test_divergence_is_detected_and_can_stop_the_run():
    guard = LossGuard(LossGuardConfig(divergence_window=50, stop_on_divergence=True,
                                      spike_sigma=1000))  # sigma high: test the trend path
    for step in range(200):
        guard.observe(step, 2.0)
    verdict = OK
    for step in range(200, 400):
        verdict = guard.observe(step, 2.0 + (step - 200) * 0.01).action
        if verdict == STOP:
            break
    assert verdict == STOP


def test_plateau_is_reported_but_never_acted_on():
    guard = LossGuard(LossGuardConfig(plateau_window=100, spike_sigma=1000))
    for step in range(400):
        assert guard.observe(step, 2.0).action == OK
    assert guard.plateau_since is not None
    assert "plateau" in guard.status()


def test_disabled_guard_does_nothing():
    guard = LossGuard(LossGuardConfig(enabled=False))
    assert guard.observe(0, float("nan")).action == OK
    assert guard.observe(1, 1e9).action == OK


# --- snapshots -------------------------------------------------------------

def test_snapshot_round_trips_the_weights():
    model = GeocentricGPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2,
                                    n_kv_head=1, n_embd=32))
    snapshot = WeightSnapshot(every=10)
    assert snapshot.maybe_take(model, 10, healthy=True)
    good = model.blocks[0].mlp.gate.weight.detach().clone()

    with torch.no_grad():
        model.blocks[0].mlp.gate.weight.add_(5.0)
    assert not torch.allclose(model.blocks[0].mlp.gate.weight, good)

    assert snapshot.restore(model) == 10
    assert torch.allclose(model.blocks[0].mlp.gate.weight, good)


def test_snapshot_refuses_unhealthy_states_and_respects_its_interval():
    model = GeocentricGPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2,
                                    n_kv_head=1, n_embd=32))
    snapshot = WeightSnapshot(every=100)
    assert not snapshot.maybe_take(model, 5, healthy=False)
    assert snapshot.step is None
    assert snapshot.maybe_take(model, 5, healthy=True)
    assert not snapshot.maybe_take(model, 50, healthy=True), "too soon"
    assert snapshot.maybe_take(model, 200, healthy=True)


def test_disabled_snapshot_costs_nothing():
    model = GeocentricGPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2,
                                    n_kv_head=1, n_embd=32))
    snapshot = WeightSnapshot(enabled=False)
    assert not snapshot.maybe_take(model, 1, healthy=True)
    assert snapshot.restore(model) is None
    assert snapshot.bytes == 0


# --- resume selection ------------------------------------------------------

def test_resume_prefers_the_best_checkpoint_when_it_is_actually_better(tmp_path):
    (tmp_path / "m_pretrained.pt").write_bytes(b"x")
    (tmp_path / "m_pretrained_best.pt").write_bytes(b"x")
    record_checkpoint(tmp_path, "m_pretrained.pt", 2000, loss=3.1, eval_loss=3.4)
    record_checkpoint(tmp_path, "m_pretrained_best.pt", 1500, loss=2.9, eval_loss=3.0)

    name, why = choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt")
    assert name == "m_pretrained_best.pt"
    assert "3.0000" in why and "held-out" in why


def test_resume_keeps_the_latest_when_it_is_already_the_best(tmp_path):
    (tmp_path / "m_pretrained.pt").write_bytes(b"x")
    (tmp_path / "m_pretrained_best.pt").write_bytes(b"x")
    record_checkpoint(tmp_path, "m_pretrained.pt", 2000, loss=2.5, eval_loss=2.7)
    record_checkpoint(tmp_path, "m_pretrained_best.pt", 1500, loss=2.9, eval_loss=3.0)
    name, _ = choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt")
    assert name == "m_pretrained.pt"


def test_resume_falls_back_to_latest_for_a_run_with_no_ledger(tmp_path):
    """A run started before checkpoints.json existed must resume exactly as it used to."""
    (tmp_path / "m_pretrained.pt").write_bytes(b"x")
    (tmp_path / "m_pretrained_best.pt").write_bytes(b"x")
    name, why = choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt")
    assert name == "m_pretrained.pt"
    assert "predates" in why


def test_resume_modes_are_honoured(tmp_path):
    (tmp_path / "m_pretrained.pt").write_bytes(b"x")
    (tmp_path / "m_pretrained_best.pt").write_bytes(b"x")
    record_checkpoint(tmp_path, "m_pretrained.pt", 2000, loss=2.5)
    record_checkpoint(tmp_path, "m_pretrained_best.pt", 1500, loss=2.9)
    assert choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt",
                                    "best")[0] == "m_pretrained_best.pt"
    assert choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt",
                                    "last")[0] == "m_pretrained.pt"


def test_resume_handles_a_missing_checkpoint(tmp_path):
    assert choose_resume_checkpoint(tmp_path, "a.pt", "b.pt")[0] is None
    (tmp_path / "b.pt").write_bytes(b"x")
    assert choose_resume_checkpoint(tmp_path, "a.pt", "b.pt")[0] == "b.pt"


def test_a_corrupt_ledger_does_not_break_resume(tmp_path):
    (tmp_path / "m_pretrained.pt").write_bytes(b"x")
    (tmp_path / "m_pretrained_best.pt").write_bytes(b"x")
    (tmp_path / "checkpoints.json").write_text("{not json", encoding="utf-8")
    name, why = choose_resume_checkpoint(tmp_path, "m_pretrained.pt", "m_pretrained_best.pt")
    assert name == "m_pretrained.pt" and "unreadable" in why


def test_an_exhausted_guard_stands_down_instead_of_stalling_the_run():
    """Skipping every update forever is a worse failure than letting them through."""
    config = LossGuardConfig(max_consecutive_spikes=2, max_rollbacks=1, rewarm_steps=1)
    guard = LossGuard(config)
    descending(guard)
    big = guard.mean + 50 * max(guard.std, 0.1)

    actions = [guard.observe(300 + i, big).action for i in range(40)]
    assert ROLLBACK in actions
    assert actions[-1] == OK, "the guard must give up, not block progress indefinitely"
    assert guard.exhausted_at is not None
    assert "stood down" in guard.status()
    # And having stood down, it re-baselines on the new regime instead of flagging forever.
    assert guard.mean > 0


def test_a_stood_down_guard_still_refuses_non_finite_losses():
    """A NaN is not recoverable at any budget; it must never reach the weights."""
    config = LossGuardConfig(max_consecutive_spikes=2, max_rollbacks=1, rewarm_steps=1)
    guard = LossGuard(config)
    descending(guard)
    big = guard.mean + 50 * max(guard.std, 0.1)
    for i in range(30):
        guard.observe(300 + i, big)
    assert guard.exhausted_at is not None
    assert guard.observe(400, float("nan")).action == SKIP
