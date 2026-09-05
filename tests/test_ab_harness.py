"""Tests for the legacy-vs-current comparison harness.

The scoring function is the part that decides the verdict, so it is tested against
hand-computed values rather than only smoke-tested.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("_ab_worker", "_ab_worker.py")
compare = _load("ab_compare", "ab_compare.py")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from geocentric.data import iter_documents
    from geocentric.model import GPTConfig, GeocentricGPT
    from geocentric.tokenizer_train import train_byte_bpe_tokenizer

    d = tmp_path_factory.mktemp("ab")
    text = " ".join(
        f"The city number {i} was founded near a river and later expanded greatly."
        for i in range(600)
    )
    (d / "c.txt").write_text(text, encoding="utf-8")
    tok = train_byte_bpe_tokenizer(iter_documents(d / "c.txt"), d / "tok.json", vocab_size=800)
    model = GeocentricGPT(GPTConfig(
        vocab_size=tok.get_vocab_size(), block_size=64, n_layer=2,
        n_head=4, n_kv_head=2, n_embd=64,
    )).eval()
    return model, tok, text


def test_bits_per_byte_matches_a_manual_computation(tiny):
    """Recompute the score independently and require the two to agree."""
    model, tok, text = tiny
    device = torch.device("cpu")
    result = worker.bits_per_byte(model, tok, text, device, stride_fraction=0.5)

    block, stride = 64, 32
    context = block - stride
    ids = tok.encode(text).ids

    total_nll = 0.0
    counted = 0
    start = 0
    with torch.no_grad():
        while start + block < len(ids):
            window = ids[start : start + block + 1]
            logits, _ = model(torch.tensor([window[:-1]]))
            logprobs = torch.log_softmax(logits[0].float(), dim=-1)
            for position in range(context, block):
                total_nll -= float(logprobs[position, window[position + 1]])
                counted += 1
            start += stride

    assert counted == result["scored_tokens"]
    assert result["nats_per_token"] == pytest.approx(total_nll / counted, rel=1e-4)
    assert result["bits_per_byte"] == pytest.approx(
        total_nll / math.log(2) / result["scored_bytes"], rel=1e-4
    )


def test_every_scored_token_gets_equal_context(tiny):
    """Unequal context between arms would bias the comparison."""
    model, tok, text = tiny
    result = worker.bits_per_byte(model, tok, text, torch.device("cpu"))
    assert result["context_per_scored_token"] == 32


def test_scored_bytes_cover_only_the_scored_span(tiny):
    """Numerator and denominator must describe the same region of text."""
    model, tok, text = tiny
    result = worker.bits_per_byte(model, tok, text, torch.device("cpu"))
    assert 0 < result["scored_bytes"] <= len(text.encode("utf-8"))
    # Scored tokens should account for most of the text, not a sliver of it.
    assert result["scored_bytes"] > 0.5 * len(text.encode("utf-8"))
    implied = result["scored_tokens"] / result["scored_bytes"]
    assert result["tokens_per_byte"] == pytest.approx(implied)


def test_an_untrained_model_scores_near_its_uniform_bound(tiny):
    """Sanity anchor: random weights cannot beat log2(vocab) per token."""
    model, tok, text = tiny
    result = worker.bits_per_byte(model, tok, text, torch.device("cpu"))
    ceiling_bits_per_token = math.log2(tok.get_vocab_size())
    actual_bits_per_token = result["nats_per_token"] / math.log(2)
    assert actual_bits_per_token <= ceiling_bits_per_token + 0.5
    assert result["bits_per_byte"] > 0


def test_short_text_is_rejected_rather_than_silently_scored(tiny):
    model, tok, _ = tiny
    with pytest.raises(ValueError, match="too short"):
        worker.bits_per_byte(model, tok, "tiny", torch.device("cpu"))


def test_kwargs_are_filtered_to_the_target_signature():
    """The legacy trainer lacks max_steps and n_kv_head; passing them must not crash."""
    def legacy_like(data_path, output_dir, epochs=1):
        return (data_path, output_dir, epochs)

    assert worker._call_filtered(
        legacy_like, data_path="d", output_dir="o", epochs=2,
        max_steps=100, n_kv_head=4,
    ) == ("d", "o", 2)


def test_report_renders_and_names_the_better_arm(tmp_path):
    results = {
        "legacy": {
            "heldout": {"bits_per_byte": 2.0, "tokens_per_byte": 0.30},
            "params": 35_000_000, "vocab_size_actual": 8192, "block_size_actual": 256,
            "train_seconds": 100.0, "tokens_per_second": 5000.0,
            "peak_vram_gb": 4.0, "peak_host_rss_gb": 12.0, "steps": 500,
            "samples": [{"prompt": "The city", "completion": "was was was"}],
        },
        "current": {
            "heldout": {"bits_per_byte": 1.5, "tokens_per_byte": 0.25},
            "params": 51_000_000, "vocab_size_actual": 32000, "block_size_actual": 1024,
            "train_seconds": 80.0, "tokens_per_second": 9000.0,
            "peak_vram_gb": 5.0, "peak_host_rss_gb": 3.0, "steps": 400,
            "samples": [{"prompt": "The city", "completion": "of Rome grew steadily."}],
        },
    }
    text = compare.report(results, tmp_path)
    assert (tmp_path / "REPORT.md").exists()
    assert "25.0% better" in text
    assert "2.0000" in text and "1.5000" in text
    assert "of Rome grew steadily." in text
    # Host RAM fell, so the current arm must be credited, not penalised.
    assert "better by 75.0%" in text


def test_report_survives_a_failed_arm(tmp_path):
    """One arm crashing must still produce a readable report."""
    results = {"current": {
        "heldout": {"bits_per_byte": 1.5, "tokens_per_byte": 0.25},
        "params": 51_000_000, "vocab_size_actual": 32000, "block_size_actual": 1024,
        "samples": [],
    }}
    text = compare.report(results, tmp_path)
    assert "—" in text
    assert "1.5000" in text


def test_capping_steps_against_the_legacy_arm_is_refused():
    """max_steps exists only in the current trainer; silently unequal data would lie."""
    argv = [
        "ab_compare.py", "--data", "x.txt", "--max_steps", "100",
        "--arms", "legacy,current",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as excinfo:
            compare.main()
        assert "max_steps" in str(excinfo.value)
    finally:
        sys.argv = old


def test_changing_the_slice_size_invalidates_the_cache(tmp_path):
    source = tmp_path / "src.txt"
    source.write_text("x" * (6 * 1024 * 1024), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    train, _ = compare.prepare_data(source, work, corpus_mb=1.0, heldout_mb=0.5)
    assert train.stat().st_size == 1024 * 1024
    # A different request must re-cut rather than reuse the previous slice.
    train, _ = compare.prepare_data(source, work, corpus_mb=2.0, heldout_mb=0.5)
    assert train.stat().st_size == 2 * 1024 * 1024
