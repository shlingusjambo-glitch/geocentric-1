from __future__ import annotations

import json

import pytest
import torch

from geocentric.parallax import probes
from geocentric.parallax.report import diagnose, render_report
from geocentric.parallax.suite import (
    ProbeResult,
    SuiteResult,
    _distinct_n,
    _longest_repeat_run,
    _weights,
    probe_meridian,
    probe_sextant,
    probe_zenith,
    score_accuracy,
    score_bpb,
    score_context_gain,
    score_distinct,
)
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.tokenizer_train import train_byte_bpe_tokenizer


@pytest.fixture
def pair(tmp_path):
    tokenizer = train_byte_bpe_tokenizer(
        [probes.MERIDIAN_DOCUMENT, *probes.ZENITH_SLICES.values()],
        tmp_path / "tokenizer.json", vocab_size=800,
    )
    model = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=128,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    )).eval()
    return model, tokenizer


# --- score curves ----------------------------------------------------------

def test_score_curves_are_monotonic_and_anchored():
    assert score_bpb(0.5) == 100 and score_bpb(3.5) == 0
    assert score_bpb(1.0) > score_bpb(1.5) > score_bpb(2.0)
    assert 50 < score_bpb(1.0) < 70, "a well-trained small model should land mid-scale"

    assert score_context_gain(-0.2) == 0 and score_context_gain(1.0) == 100
    assert score_accuracy(0.25, 0.25) == 0 and score_accuracy(1.0, 0.25) == 100
    assert score_distinct(0.1) == 0 and score_distinct(0.9) == 100


def test_score_bpb_survives_nonsense_input():
    assert score_bpb(0.0) == 0.0
    assert score_bpb(float("nan")) == 0.0
    assert score_bpb(float("inf")) == 0.0


# --- degeneracy helpers ----------------------------------------------------

def test_distinct_n_detects_a_loop():
    assert _distinct_n(list(range(50)), 3) == 1.0
    assert _distinct_n([1, 2] * 25, 3) < 0.1


def test_longest_repeat_run_finds_the_cycle():
    assert _longest_repeat_run([1, 2, 3] * 8) >= 5
    assert _longest_repeat_run(list(range(40))) == 0


# --- probes run and produce sane shapes ------------------------------------

def test_zenith_scores_every_slice(pair):
    model, tokenizer = pair
    result = probe_zenith(model, tokenizer, torch.device("cpu"), 128, probes.ZENITH_SLICES)
    assert result.ran and 0 <= result.score <= 100
    assert set(result.detail["by_slice"]) == set(probes.ZENITH_SLICES)
    # An untrained model is near the uniform-byte ceiling, so it must score near zero.
    assert result.score < 20
    assert result.detail["bits_per_byte"] > 1.5


def test_meridian_returns_a_bucket_curve(pair):
    model, tokenizer = pair
    result = probe_meridian(model, tokenizer, torch.device("cpu"), 128, probes.MERIDIAN_DOCUMENT)
    assert result.ran
    curve = result.detail["curve_nats"]
    assert len([c for c in curve if c is not None]) >= 2
    assert result.detail["gain_nats"] == pytest.approx(
        curve[0] - [c for c in curve if c is not None][-1], abs=1e-6
    )


def test_meridian_refuses_a_document_it_cannot_bucket(pair):
    model, tokenizer = pair
    result = probe_meridian(model, tokenizer, torch.device("cpu"), 128, "short")
    assert not result.ran and "too short" in result.note


def test_sextant_is_near_chance_for_an_untrained_model(pair):
    model, tokenizer = pair
    result = probe_sextant(model, tokenizer, torch.device("cpu"), 128, probes.SEXTANT_ITEMS)
    assert result.ran
    assert result.detail["chance"] == pytest.approx(0.25)
    assert result.detail["accuracy"] < 0.6, "random weights should not know things"


def test_every_sextant_item_has_three_distractors():
    for prefix, answer, distractors in probes.SEXTANT_ITEMS:
        assert len(distractors) == 3, prefix
        assert answer not in distractors
        assert answer.startswith(" "), f"{answer!r} needs a leading space to tokenize cleanly"


def test_astrolabe_checks_are_all_implemented():
    from geocentric.parallax.suite import _check

    for item in probes.ASTROLABE_ITEMS:
        _check(item, "some answer", True)  # must not raise


# --- index weighting -------------------------------------------------------

def test_a_skipped_probe_is_excluded_rather_than_counted_as_zero():
    full = _weights("pretrained", ["ZENITH", "MERIDIAN", "SEXTANT", "NADIR"])
    partial = _weights("pretrained", ["ZENITH", "SEXTANT"])
    assert sum(full.values()) == pytest.approx(1.0)
    assert sum(partial.values()) == pytest.approx(1.0)
    assert "MERIDIAN" not in partial
    assert partial["ZENITH"] > full["ZENITH"]


def test_sft_weighting_puts_instruction_following_first():
    weights = _weights("sft", ["ZENITH", "MERIDIAN", "SEXTANT", "ASTROLABE", "NADIR"])
    assert max(weights, key=weights.get) == "ASTROLABE"


def test_prism_joins_the_index_when_it_runs():
    full = ["ZENITH", "MERIDIAN", "SEXTANT", "ASTROLABE", "NADIR"]
    without = _weights("sft", full)
    with_prism = _weights("sft", [*full, "PRISM"])
    assert "PRISM" not in without
    assert with_prism["PRISM"] == pytest.approx(0.20, abs=0.01)
    # The text probes give up exactly PRISM's share between them, in proportion.
    assert with_prism["ASTROLABE"] == pytest.approx(without["ASTROLABE"] * 0.8, abs=0.01)
    assert sum(with_prism.values()) == pytest.approx(1.0, abs=0.01)


def test_prism_takes_a_bigger_share_when_the_text_probes_are_skipped():
    """Renormalizing is the point: whatever ran carries the whole index between it."""
    weights = _weights("sft", ["ZENITH", "ASTROLABE", "PRISM"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["PRISM"] > 0.20


# --- diagnosis -------------------------------------------------------------

def _suite(**kwargs) -> SuiteResult:
    base = dict(
        model_dir="runs/x", model_name="T", stage="pretrained", params=100_000_000,
        block_size=1024, vocab_size=32000, watermark=None, multimodal=False, device="cpu",
        probes=[ProbeResult("ZENITH", 50.0, "1.2 bits/byte", {"spread": 0.1}),
                ProbeResult("MERIDIAN", 80.0, "+0.3 nats", {"gain_nats": 0.3})],
        index=60.0, weights={"ZENITH": 0.6, "MERIDIAN": 0.4}, training={}, seconds=1.0,
    )
    base.update(kwargs)
    return SuiteResult(**base)


def test_diagnosis_names_the_token_budget_when_the_run_was_starved():
    result = _suite(training={
        "tokens_seen": 100_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
    })
    findings = diagnose(result)
    top = findings[0]
    assert top["severity"] == "critical"
    assert "budget" in top["title"].lower()
    assert "20.0x short" in top["evidence"]


def test_diagnosis_flags_a_run_that_stopped_while_still_learning():
    result = _suite(training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
        "history": {"steps": list(range(0, 1000, 50)),
                    "loss": [5.0 - i * 0.1 for i in range(20)]},
    })
    titles = [f["title"] for f in diagnose(result)]
    assert any("still falling" in t for t in titles)


def test_diagnosis_flags_overfitting():
    result = _suite(training={"loss": 1.0, "eval_loss": 2.0,
                              "tokens_seen": 2_000_000_000,
                              "config": {"recommended_tokens": 2_000_000_000}})
    titles = [f["title"] for f in diagnose(result)]
    assert any("Validation loss" in t for t in titles)


def test_diagnosis_explains_a_dead_context():
    result = _suite(probes=[ProbeResult("MERIDIAN", 0.0, "-0.01 nats", {"gain_nats": -0.01})])
    finding = next(f for f in diagnose(result) if "context" in f["title"])
    assert finding["severity"] == "critical"
    assert "doc_sep" in finding["fix"]


def test_diagnosis_flags_a_low_sft_learning_rate():
    result = _suite(stage="sft", training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000, "learning_rate": 2e-5},
    })
    finding = next(f for f in diagnose(result) if "rate" in f["title"])
    assert finding["severity"] == "critical"


def test_diagnosis_always_says_something():
    assert diagnose(_suite(training={})), "an empty training record must still produce a finding"


def test_report_renders_every_section():
    text = render_report(_suite())
    for heading in ("# PARALLAX report", "## Scores", "## Strengths", "## Weaknesses",
                    "## Why — diagnosis", "## Training record", "## Probe detail",
                    "## Reproducing this"):
        assert heading in text, heading
    assert "Parallax Index" in text


def test_report_writes_markdown_and_json(tmp_path):
    from geocentric.parallax import write_report

    path = write_report(_suite(), tmp_path / "PARALLAX.md")
    assert path.exists()
    data = json.loads(path.with_suffix(".json").read_text())
    assert data["index"] == 60.0
    assert [p["name"] for p in data["probes"]] == ["ZENITH", "MERIDIAN"]


def test_diagnosis_explains_a_rollback():
    result = _suite(training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
        "loss_guard": {"spikes": 9, "rollbacks": 2, "skipped_updates": 7,
                       "rollback_steps": [4100, 8300]},
    })
    finding = next(f for f in diagnose(result) if "roll back" in f["title"])
    assert "4,100" in finding["evidence"] and "8,300" in finding["evidence"]


def test_diagnosis_reports_absorbed_spikes_as_minor():
    result = _suite(training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
        "loss_guard": {"spikes": 9, "rollbacks": 0, "skipped_updates": 9},
    })
    finding = next(f for f in diagnose(result) if "spike" in f["title"].lower())
    assert finding["severity"] == "minor"


def test_diagnosis_explains_a_plateau():
    result = _suite(training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
        "loss_guard": {"plateau_since_step": 5000, "best_loss": 2.1, "best_step": 5000},
    })
    finding = next(f for f in diagnose(result) if "stopped improving" in f["title"])
    assert "5,000" in finding["evidence"]


def test_a_disabled_epicycle_reports_as_off():
    """A config with enabled=False still carries each gear's default; do not print those."""
    from geocentric.epicycle import EpicycleConfig
    from geocentric.parallax.report import _epicycle_gears

    off = EpicycleConfig.preset("off").to_dict()
    assert off["deferent"] is True, "the default is still set; that is what makes this a trap"
    assert _epicycle_gears(off) == "off"
    assert _epicycle_gears(None) == "off"
    assert _epicycle_gears(EpicycleConfig.preset("speed").to_dict()) == "deferent, horizon"


def test_the_training_record_reports_epicycle_honestly():
    result = _suite(training={"config": {"epicycle": {"enabled": False, "deferent": True,
                                                      "horizon": True, "equant": True}}})
    assert "| EPICYCLE | off |" in render_report(result)


# --- LODESTAR --------------------------------------------------------------

def _watermarked(tmp_path, delta=2.0):
    from geocentric.watermark import WatermarkConfig

    tokenizer = train_byte_bpe_tokenizer(
        [probes.MERIDIAN_DOCUMENT * 3], tmp_path / "tokenizer.json", vocab_size=900,
    )
    model = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=256,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    )).eval()
    model.config.watermark = WatermarkConfig(identity="Probe Subject", delta=delta).to_dict()
    return model, tokenizer


def test_lodestar_skips_an_unwatermarked_model(pair):
    from geocentric.parallax.suite import probe_lodestar

    model, tokenizer = pair
    result = probe_lodestar(model, tokenizer, torch.device("cpu"), probes.NADIR_PROMPTS[:2],
                            "pretrained")
    assert not result.ran and "not watermarked" in result.note


def test_lodestar_measures_detectability_and_cost(tmp_path):
    from geocentric.parallax.suite import probe_lodestar

    model, tokenizer = _watermarked(tmp_path, delta=4.0)
    result = probe_lodestar(model, tokenizer, torch.device("cpu"), probes.NADIR_PROMPTS[:3],
                            "pretrained", max_tokens=96)
    assert result.ran, result.note
    detail = result.detail
    # The mark must be found under its own name and nowhere else.
    assert detail["mean_z_true_identity"] > 4
    assert abs(detail["mean_z_decoy_identity"]) < 3
    assert abs(detail["mean_z_unmarked_text"]) < 3
    # And the cost of the bias must be measured, not assumed to be zero.
    assert detail["quality_cost_nats_per_token"] > 0


def test_lodestar_refuses_to_decide_on_short_replies(tmp_path):
    from geocentric.parallax.suite import probe_lodestar

    model, tokenizer = _watermarked(tmp_path)
    result = probe_lodestar(model, tokenizer, torch.device("cpu"), probes.NADIR_PROMPTS[:2],
                            "pretrained", max_tokens=8)
    assert not result.ran
    assert "too short" in result.headline


def test_diagnosis_flags_a_weak_or_expensive_watermark():
    weak = _suite(probes=[ProbeResult("LODESTAR", 20.0, "z=2",
                                      {"mean_z_true_identity": 2.0, "delta": 2.0,
                                       "quality_cost_nats_per_token": 0.01,
                                       "mean_generation_tokens": 100})])
    assert any("not reliably detectable" in f["title"] for f in diagnose(weak))

    costly = _suite(probes=[ProbeResult("LODESTAR", 40.0, "z=9",
                                        {"mean_z_true_identity": 9.0, "delta": 6.0,
                                         "quality_cost_nats_per_token": 0.4,
                                         "mean_generation_tokens": 100})])
    assert any("degrading output" in f["title"] for f in diagnose(costly))


def test_diagnosis_flags_an_unusable_tokenizer():
    result = _suite(probes=[ProbeResult("ZENITH", 0.0, "8.0 bits/byte",
                                        {"unk_rate": 0.05, "spread": 0.1})],
                    vocab_size=700)
    finding = next(f for f in diagnose(result) if "tokenizer cannot represent" in f["title"])
    assert finding["severity"] == "critical"
    assert "5.0%" in finding["evidence"]


def test_meridian_samples_early_buckets_from_more_than_the_documents_opening(pair):
    """With context-skipping on, bucket 0 would only ever see the first sentence."""
    from geocentric.parallax.suite import _token_losses

    model, tokenizer = pair
    ids = tokenizer.encode(probes.MERIDIAN_DOCUMENT).ids
    _, skipped = _token_losses(model, ids, torch.device("cpu"), 128, stride=32,
                               skip_context=True)
    _, complete = _token_losses(model, ids, torch.device("cpu"), 128, stride=32,
                                skip_context=False)
    early_skipped = int((skipped < 16).sum())
    early_complete = int((complete < 16).sum())
    assert early_complete > early_skipped * 3, (early_complete, early_skipped)


def test_diagnosis_flags_a_guard_that_stood_down():
    result = _suite(training={
        "tokens_seen": 2_000_000_000,
        "config": {"recommended_tokens": 2_000_000_000},
        "loss_guard": {"spikes": 40, "rollbacks": 3, "exhausted_at_step": 12000,
                       "rollback_steps": [4000, 8000, 11000]},
    })
    finding = next(f for f in diagnose(result) if "stood down" in f["title"])
    assert finding["severity"] == "critical"
    assert "resume_from best" in finding["fix"]


def test_a_probe_that_did_not_run_produces_no_verdict_about_the_thing_it_measures():
    """LODESTAR skipping must not read as 'your watermark is weak'."""
    result = _suite(probes=[ProbeResult("LODESTAR", None, "too short to decide",
                                        {"mean_generation_tokens": 12, "minimum_tokens": 40,
                                         "identity": "X", "delta": 2.0})])
    findings = diagnose(result)
    assert not any("not reliably detectable" in f["title"] for f in findings)
    finding = next(f for f in findings if "could not be measured" in f["title"])
    assert finding["severity"] == "info"
