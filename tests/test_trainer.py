from __future__ import annotations

import torch

from geocentric.param_compiler import compute_fluid_dimensions, exact_params, recommended_tokens
from geocentric.trainer import build_optimizer, lr_at_step
from geocentric.model import GPTConfig, GeocentricGPT


def test_warmup_then_cosine_decay():
    total, warm, peak = 1000, 100, 1e-3
    assert lr_at_step(0, total, peak, warm) < peak
    assert abs(lr_at_step(warm - 1, total, peak, warm) - peak) < 1e-9
    mid = lr_at_step(total // 2, total, peak, warm)
    assert 0.1 * peak < mid < peak
    assert lr_at_step(total, total, peak, warm, 0.1) == peak * 0.1
    # monotonically decreasing after warmup
    values = [lr_at_step(s, total, peak, warm) for s in range(warm, total, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_only_matrices_are_weight_decayed():
    model = GeocentricGPT(GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, n_embd=64, block_size=32))
    optimizer = build_optimizer(model, 1e-3, weight_decay=0.1, device_type="cpu")
    decayed, undecayed = optimizer.param_groups[0], optimizer.param_groups[1]
    assert decayed["weight_decay"] == 0.1
    assert undecayed["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() < 2 for p in undecayed["params"])


def test_planner_hits_its_parameter_target():
    for preset in ["50m", "120m", "250m", "350m"]:
        dims = compute_fluid_dimensions(preset, vocab_size=32000)
        target = float(preset[:-1]) * 1e6
        assert abs(dims["params"] - target) / target < 0.15
        assert dims["n_embd"] % dims["n_head"] == 0
        assert dims["n_head"] % dims["n_kv_head"] == 0
        # A 256-token context was the old default and is far too short for coherence.
        assert dims["block_size"] >= 1024


def test_planner_matches_the_real_model_size():
    dims = compute_fluid_dimensions("120m", vocab_size=32000)
    model = GeocentricGPT(GPTConfig(
        vocab_size=32000, block_size=dims["block_size"], n_layer=dims["n_layer"],
        n_head=dims["n_head"], n_kv_head=dims["n_kv_head"], n_embd=dims["n_embd"],
    ))
    predicted = exact_params(32000, dims["n_layer"], dims["n_embd"], dims["n_head"], dims["n_kv_head"])
    assert predicted == model.num_params()


def test_token_budget_follows_chinchilla():
    assert recommended_tokens(250_000_000) == 5_000_000_000
