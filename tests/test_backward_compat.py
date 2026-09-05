"""A checkpoint trained before any of this existed must still load, resume and run.

These tests build checkpoints in the *old* shape by hand rather than by calling the
current save path, so they keep failing if a future change quietly starts depending
on a field that older files do not have.
"""
from __future__ import annotations

import json

import torch

from geocentric.checkpoint import (
    load_checkpoint,
    load_optimizer_state,
    pretrained_checkpoint_name,
)
from geocentric.model import GPTConfig, GeocentricGPT

# Exactly the fields GPTConfig carried before watermarking and vision were added.
LEGACY_CONFIG_FIELDS = [
    "vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "n_embd",
    "dropout", "bias", "rope_theta", "norm_eps", "model_name", "gradient_checkpointing",
]


def legacy_model():
    return GeocentricGPT(GPTConfig(
        vocab_size=256, block_size=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=64,
    ))


def write_legacy_checkpoint(path, model, step: int = 500, with_optimizer: bool = True):
    """A payload in the pre-change format: no watermark, no vision, no loss in extra."""
    optimizer_state = None
    if with_optimizer:
        # Stepped before the payload is built, so the saved weights and the live model
        # agree without relying on state_dict() returning references.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.randint(0, 256, (1, 16)), labels=torch.randint(0, 256, (1, 16)))[1].backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        optimizer_state = optimizer.state_dict()

    config = {k: getattr(model.config, k) for k in LEGACY_CONFIG_FIELDS}
    payload = {"model": {k: v.clone() for k, v in model.state_dict().items()},
               "config": config, "step": step, "stage": "pretrained"}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, path)
    return config


def test_a_legacy_checkpoint_loads(tmp_path):
    model = legacy_model()
    name = pretrained_checkpoint_name("legacy")
    write_legacy_checkpoint(tmp_path / name, model)

    restored = load_checkpoint(tmp_path, device=torch.device("cpu"), checkpoint_name=name)
    assert restored.config.watermark is None
    assert restored.config.vision is None
    assert restored.vision is None
    assert restored.active_layers is None


def test_a_legacy_checkpoint_produces_identical_logits(tmp_path):
    """Not just "loads" — the same numbers. Nothing added may perturb the forward pass."""
    model = legacy_model().eval()
    ids = torch.randint(0, 256, (2, 32))
    name = pretrained_checkpoint_name("legacy")
    # Written first: the helper takes a real optimizer step, which moves the weights.
    write_legacy_checkpoint(tmp_path / name, model)
    with torch.no_grad():
        expected, _ = model(ids)

    restored = load_checkpoint(tmp_path, device=torch.device("cpu"), checkpoint_name=name).eval()
    with torch.no_grad():
        actual, _ = restored(ids)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_a_legacy_config_json_loads(tmp_path):
    model = legacy_model()
    (tmp_path / "config.json").write_text(
        json.dumps({k: getattr(model.config, k) for k in LEGACY_CONFIG_FIELDS}), encoding="utf-8"
    )
    config = GPTConfig.load(tmp_path / "config.json")
    assert config.watermark is None and config.vision is None
    assert config.n_layer == 4 and config.n_embd == 64


def test_a_legacy_optimizer_state_still_restores(tmp_path):
    model = legacy_model()
    name = pretrained_checkpoint_name("legacy")
    write_legacy_checkpoint(tmp_path / name, model, step=1234)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    assert load_optimizer_state(tmp_path, name, optimizer) == 1234
    assert optimizer.state, "Adam moments were not restored"


def test_resume_from_a_legacy_run_takes_the_latest_checkpoint(tmp_path):
    """No checkpoints.json means no basis for preferring the best — behave as before."""
    from geocentric.loss_guard import choose_resume_checkpoint

    model = legacy_model()
    last, best = pretrained_checkpoint_name("legacy"), pretrained_checkpoint_name("legacy", best=True)
    write_legacy_checkpoint(tmp_path / last, model, step=900)
    write_legacy_checkpoint(tmp_path / best, model, step=500)
    for mode in ("auto", "last"):
        assert choose_resume_checkpoint(tmp_path, last, best, mode)[0] == last
    # And an explicit request is still honoured.
    assert choose_resume_checkpoint(tmp_path, last, best, "best")[0] == best


def test_generation_from_a_legacy_checkpoint_is_unwatermarked(tmp_path):
    from geocentric.generate import make_processor

    model = legacy_model()
    name = pretrained_checkpoint_name("legacy")
    write_legacy_checkpoint(tmp_path / name, model)
    restored = load_checkpoint(tmp_path, device=torch.device("cpu"), checkpoint_name=name)
    assert make_processor(restored) is None
    out = restored.generate(torch.randint(0, 256, (1, 4)), max_new_tokens=8, temperature=0.0)
    assert out.size(1) == 12


def test_a_legacy_metrics_file_survives_an_update(tmp_path):
    """training_metrics.json from an older run has no history.elapsed list."""
    from geocentric.training_metrics import load_training_metrics, update_training_metrics

    (tmp_path / "training_metrics.json").write_text(json.dumps({
        "phase": "pretraining", "status": "running", "config": {"total_steps": 100},
        "step": 10, "loss": 4.0, "start_time": "2024-01-01T00:00:00",
        "history": {"steps": [10], "loss": [4.0], "eval_loss": []},
    }), encoding="utf-8")

    update_training_metrics(tmp_path, {"step": 20, "loss": 3.5})
    metrics = load_training_metrics(tmp_path)
    assert metrics["history"]["steps"] == [10, 20]
    assert len(metrics["history"]["elapsed"]) == 1  # added going forward, not backfilled


def test_a_legacy_run_benchmarks_without_a_training_record(tmp_path):
    """PARALLAX must degrade to score-only analysis, not crash, when metrics are absent."""
    from geocentric.parallax.report import diagnose, render_report
    from geocentric.parallax.suite import ProbeResult, SuiteResult

    result = SuiteResult(
        model_dir=str(tmp_path), model_name="Legacy", stage="pretrained", params=120_000_000,
        block_size=1024, vocab_size=32000, watermark=None, multimodal=False, device="cpu",
        probes=[ProbeResult("ZENITH", 55.0, "1.1 bits/byte", {"spread": 0.2})],
        index=55.0, weights={"ZENITH": 1.0}, training={}, seconds=1.0,
    )
    assert diagnose(result)
    text = render_report(result)
    assert "No `training_metrics.json` was found" in text


def test_epicycle_off_leaves_the_training_path_unchanged():
    """The default must be the code path a pre-change run took."""
    from geocentric.epicycle import EpicycleConfig, EpicycleScheduler

    model = legacy_model()
    config = EpicycleConfig.preset("off")
    assert not config.enabled
    sched = EpicycleScheduler(config, model, total_steps=1000, block_size=128)
    assert model.active_layers is None
    for step in (0, 500, 1000):
        assert sched.apply(step) == (4, 128)
        assert sched.status(step) == ""


def test_the_default_forward_signature_is_backward_compatible():
    """Positional calls made by older code must still mean what they meant."""
    model = legacy_model().eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        a = model(ids)
        b = model(ids, None)                       # labels positional
        c = model(ids, labels=None, attention_mask=None, caches=None, position_offset=0)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[0], c[0])
    assert a[1] is None
