from __future__ import annotations

import torch

from geocentric.epicycle import (
    EpicycleConfig,
    EpicycleScheduler,
    RingAdamW,
    equant_loss,
)
from geocentric.model import GPTConfig, GeocentricGPT


def build(n_layer=8, block_size=1024):
    return GeocentricGPT(GPTConfig(
        vocab_size=256, block_size=block_size, n_layer=n_layer,
        n_head=4, n_kv_head=2, n_embd=64,
    ))


# --- DEFERENT --------------------------------------------------------------

def test_deferent_starts_shallow_and_reaches_full_depth():
    model = build(n_layer=8)
    sched = EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    assert sched.active_layers(0) == 4
    assert sched.active_layers(1000) == 8
    assert model.active_layers == 4


def test_growing_a_layer_does_not_change_the_models_output():
    """The whole point of identity insertion: no loss spike at a growth event."""
    model = build(n_layer=8)
    model.eval()
    sched = EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    ids = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        before, _ = model(ids)
        sched.apply(300)  # crosses at least one growth boundary
        assert model.active_layers is not None and model.active_layers > 4
        after, _ = model(ids)
    assert torch.allclose(before, after, atol=1e-5)


def test_inactive_layers_receive_no_gradient():
    model = build(n_layer=8)
    EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    ids = torch.randint(0, 256, (2, 16))
    _, loss = model(ids, labels=ids)
    loss.backward()
    assert model.blocks[0].mlp.gate.weight.grad is not None
    # Skipped in the forward pass, so .grad stays None and AdamW leaves them alone,
    # weight decay included.
    assert model.blocks[7].mlp.gate.weight.grad is None


def test_disabled_epicycle_leaves_the_model_at_full_depth():
    model = build(n_layer=6)
    sched = EpicycleScheduler(EpicycleConfig(enabled=False), model, total_steps=100, block_size=1024)
    assert sched.active_layers(0) == 6
    assert model.active_layers is None


# --- HORIZON ---------------------------------------------------------------

def test_horizon_ramps_context_in_powers_of_two():
    model = build(block_size=1024)
    sched = EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    lengths = [sched.context_length(s) for s in range(0, 1001, 50)]
    assert lengths[0] == 256
    assert lengths[-1] == 1024
    assert all(l & (l - 1) == 0 for l in lengths), "every context must be a power of two"
    assert lengths == sorted(lengths), "context must never shrink"


def test_crop_matches_the_requested_context():
    model = build(block_size=1024)
    sched = EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    ids = torch.randint(0, 256, (2, 1024))
    cropped, labels = sched.crop(ids, ids, 256)
    assert cropped.shape == (2, 256) and labels.shape == (2, 256)


# --- EQUANT ----------------------------------------------------------------

def test_equant_keeps_the_hard_band_and_drops_the_outliers():
    # 100 easy tokens, 98 hard ones, 2 absurd ones standing in for corrupt text.
    losses = torch.cat([torch.full((100,), 0.1), torch.full((98,), 3.0), torch.full((2,), 500.0)])
    labels = torch.zeros(200, dtype=torch.long)
    out = equant_loss(losses, labels, keep=0.49, trim=0.01)
    # 2 trimmed off the top leaves the 3.0 band; the 500s must not survive.
    assert abs(float(out) - 3.0) < 1e-4


def test_equant_raises_the_loss_above_the_plain_mean():
    losses = torch.rand(1000) * 5
    labels = torch.zeros(1000, dtype=torch.long)
    assert float(equant_loss(losses, labels)) > float(losses.mean())


def test_equant_ignores_masked_positions():
    losses = torch.tensor([1.0, 99.0, 1.0, 99.0])
    labels = torch.tensor([5, -100, 5, -100])
    assert abs(float(equant_loss(losses, labels, keep=1.0, trim=0.0)) - 1.0) < 1e-6


def test_equant_keeps_the_gradient_path():
    losses = (torch.rand(500) * 3).requires_grad_(True)
    labels = torch.zeros(500, dtype=torch.long)
    equant_loss(losses, labels).backward()
    assert losses.grad is not None and float(losses.grad.abs().sum()) > 0


# --- ARMILLARY -------------------------------------------------------------

def test_ring_adamw_reduces_optimizer_state():
    model = build(n_layer=4)
    params = list(model.parameters())
    n = sum(p.numel() for p in params)

    ring = RingAdamW(params, lr=1e-3, rings=4, dwell=2)
    plain = torch.optim.AdamW(params, lr=1e-3)
    for _ in range(3):
        loss = model(torch.randint(0, 256, (2, 16)), labels=torch.randint(0, 256, (2, 16)))[1]
        loss.backward()
        ring.step()
        plain.step()
        model.zero_grad(set_to_none=True)

    plain_bytes = sum(
        t.numel() * t.element_size()
        for s in plain.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0
    )
    assert ring.state_bytes() < plain_bytes * 0.65, (ring.state_bytes(), plain_bytes)
    # 4 bytes for v in bf16... no: 2 for v, plus 4/rings for the hot ring's momentum.
    assert ring.state_bytes() / n < 4.0


def test_ring_adamw_rotates_momentum():
    model = build(n_layer=4)
    opt = RingAdamW(model.parameters(), lr=1e-3, rings=4, dwell=1)
    seen = set()
    for _ in range(8):
        loss = model(torch.randint(0, 256, (2, 16)), labels=torch.randint(0, 256, (2, 16)))[1]
        loss.backward()
        seen.add(opt.hot_ring)
        opt.step()
        model.zero_grad(set_to_none=True)
    assert seen == {0, 1, 2, 3}
    # Exactly one ring holds momentum at a time.
    with_m = sum(1 for s in opt.state.values() if "m" in s)
    total = len(opt.state)
    assert 0 < with_m < total


def test_ring_adamw_balances_rings_by_size():
    model = build(n_layer=4)
    opt = RingAdamW(model.parameters(), lr=1e-3, rings=4)
    load = opt.ring_load
    assert min(load) > 0
    # Round-robin would drop the whole embedding matrix into one ring; greedy
    # balancing must not.
    assert max(load) / min(load) < 3.0


def test_ring_adamw_actually_reduces_the_loss():
    torch.manual_seed(0)
    model = build(n_layer=2)
    ids = torch.randint(0, 256, (4, 32))
    opt = RingAdamW(model.parameters(), lr=3e-3, rings=2, dwell=5)
    first = last = None
    for i in range(30):
        _, loss = model(ids, labels=ids)
        loss.backward()
        opt.step()
        model.zero_grad(set_to_none=True)
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first * 0.9, (first, last)


def test_ring_adamw_skips_params_without_gradients():
    """A DEFERENT-inactive block must not be decayed toward zero while it waits."""
    model = build(n_layer=8)
    EpicycleScheduler(EpicycleConfig(enabled=True), model, total_steps=1000, block_size=1024)
    opt = RingAdamW(model.parameters(), lr=1e-2, weight_decay=0.5, rings=2, dwell=1)
    sleeping = model.blocks[7].mlp.gate.weight
    before = sleeping.detach().clone()
    for _ in range(5):
        model(torch.randint(0, 256, (2, 16)), labels=torch.randint(0, 256, (2, 16)))[1].backward()
        opt.step()
        model.zero_grad(set_to_none=True)
    assert torch.equal(sleeping, before)


def test_the_advertised_memory_ratio_matches_what_is_allocated():
    """The 1.45x claim has to survive contact with the allocator."""
    model = build(n_layer=6)
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    ids = torch.randint(0, 256, (2, 16))

    def state_bytes(optimizer, steps):
        for _ in range(steps):
            model(ids, labels=ids)[1].backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
        return sum(
            t.numel() * t.element_size()
            for s in optimizer.state.values() for t in s.values()
            if torch.is_tensor(t) and t.dim() > 0
        )

    plain = state_bytes(torch.optim.AdamW(params, lr=1e-3), 3) / n
    ring = state_bytes(RingAdamW(params, lr=1e-3, rings=4, dwell=1), 6) / n

    assert abs(plain - 8.0) < 0.2, f"fp32 AdamW should hold 8 bytes/param of state, got {plain}"
    assert abs(ring - 3.0) < 0.4, f"4 rings should hold ~3 bytes/param of state, got {ring}"
    # The headline figure counts the weight and the gradient too, because a memory
    # budget holds those as well.
    ratio = (8 + plain) / (8 + ring)
    assert 1.35 < ratio < 1.55, ratio
