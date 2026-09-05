from __future__ import annotations

import math

import pytest
import torch

from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.watermark import (
    MIN_SCORED_TOKENS,
    WatermarkConfig,
    WatermarkProcessor,
    detect_ids,
)

VOCAB = 512


def model():
    torch.manual_seed(0)
    return GeocentricGPT(GPTConfig(
        vocab_size=VOCAB, block_size=256, n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    )).eval()


def test_identity_is_required():
    with pytest.raises(ValueError):
        WatermarkConfig(identity="   ")


def test_different_identities_give_different_keys():
    assert WatermarkConfig(identity="Alice").key != WatermarkConfig(identity="Bob").key
    assert WatermarkConfig(identity="Alice").key == WatermarkConfig(identity="Alice").key


def test_config_round_trips_through_a_directory(tmp_path):
    original = WatermarkConfig(identity="Round Trip", gamma=0.4, delta=3.0, context_width=2)
    original.save(tmp_path)
    loaded = WatermarkConfig.load(tmp_path)
    assert loaded == original
    assert WatermarkConfig.load(tmp_path / "nope") is None


def test_watermarked_generation_is_detectable():
    m = model()
    config = WatermarkConfig(identity="Detectable Model")
    out = m.generate(
        torch.randint(0, VOCAB, (1, 8)), max_new_tokens=200, temperature=1.0,
        logits_processor=WatermarkProcessor(config, VOCAB),
    )
    result = detect_ids(out[0].tolist(), config, VOCAB)
    assert result.watermarked
    assert result.z_score > 6
    assert result.green_fraction > 0.5


def test_unwatermarked_generation_is_not_flagged():
    m = model()
    out = m.generate(torch.randint(0, VOCAB, (1, 8)), max_new_tokens=200, temperature=1.0)
    result = detect_ids(out[0].tolist(), WatermarkConfig(identity="Nobody"), VOCAB)
    assert not result.watermarked
    assert abs(result.z_score) < 4


def test_the_wrong_identity_reads_as_noise():
    """The point of keying on identity: it attributes, not just flags."""
    m = model()
    mine = WatermarkConfig(identity="My Model")
    out = m.generate(
        torch.randint(0, VOCAB, (1, 8)), max_new_tokens=250, temperature=1.0,
        logits_processor=WatermarkProcessor(mine, VOCAB),
    )
    ids = out[0].tolist()
    assert detect_ids(ids, mine, VOCAB).z_score > 6
    for other in ("Their Model", "my model", "My  Model", "Anonymous"):
        assert detect_ids(ids, WatermarkConfig(identity=other), VOCAB).z_score < 4


def test_surrounding_whitespace_in_an_identity_is_ignored():
    """A pasted name with a trailing space must not silently fail to detect."""
    m = model()
    mine = WatermarkConfig(identity="Spaced Out")
    out = m.generate(
        torch.randint(0, VOCAB, (1, 8)), max_new_tokens=200, temperature=1.0,
        logits_processor=WatermarkProcessor(mine, VOCAB),
    )
    assert detect_ids(out[0].tolist(), WatermarkConfig(identity="  Spaced Out \n"), VOCAB).watermarked


def test_detect_best_ranks_the_true_author_first():
    m = model()
    mine = WatermarkConfig(identity="Author B")
    out = m.generate(
        torch.randint(0, VOCAB, (1, 8)), max_new_tokens=200, temperature=1.0,
        logits_processor=WatermarkProcessor(mine, VOCAB),
    )
    ids = out[0].tolist()
    candidates = [WatermarkConfig(identity=n) for n in ("Author A", "Author B", "Author C")]
    ranked = sorted(
        (detect_ids(ids, c, VOCAB) for c in candidates), key=lambda r: r.z_score, reverse=True
    )
    assert ranked[0].identity == "Author B"
    assert ranked[0].z_score > ranked[1].z_score + 4


def test_short_text_is_undecidable_rather_than_negative():
    config = WatermarkConfig(identity="Short")
    result = detect_ids(list(range(10)), config, VOCAB)
    assert not result.watermarked
    assert result.scored_tokens < MIN_SCORED_TOKENS
    assert "need" in result.reason


def test_zero_delta_leaves_logits_alone():
    config = WatermarkConfig(identity="Silent", delta=0.0)
    processor = WatermarkProcessor(config, VOCAB)
    logits = torch.randn(1, VOCAB)
    assert torch.equal(processor(logits.clone(), torch.zeros(1, 4, dtype=torch.long)), logits)


def test_green_mask_is_deterministic_and_about_gamma():
    config = WatermarkConfig(identity="Stable", gamma=0.25)
    processor = WatermarkProcessor(config, 20000)
    a = processor._mask((7,), torch.device("cpu"))
    b = processor._mask((7,), torch.device("cpu"))
    c = processor._mask((8,), torch.device("cpu"))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert abs(float(a.float().mean()) - 0.25) < 0.02


def test_the_model_config_carries_the_mark_into_generation():
    """A watermark you have to remember to pass is a watermark that gets forgotten."""
    from geocentric.generate import make_processor

    m = model()
    assert make_processor(m) is None
    m.config.watermark = WatermarkConfig(identity="Baked In").to_dict()
    processor = make_processor(m)
    assert processor is not None and processor.config.identity == "Baked In"


def test_a_stronger_delta_raises_the_z_score():
    m = model()
    ids = torch.randint(0, VOCAB, (1, 8))
    zs = []
    for delta in (1.0, 4.0):
        torch.manual_seed(3)
        config = WatermarkConfig(identity="Strength", delta=delta)
        out = m.generate(ids, max_new_tokens=180, temperature=1.0,
                         logits_processor=WatermarkProcessor(config, VOCAB))
        zs.append(detect_ids(out[0].tolist(), config, VOCAB).z_score)
    assert zs[1] > zs[0]


# --- capacity: why a watermark sometimes does nothing ----------------------

def _tokenizer(tmp_path, vocab=600):
    from geocentric.tokenizer_train import train_byte_bpe_tokenizer

    text = ("the keeper walked along the harbour and counted the boats before the fog "
            "returned to the valley near the bridge ") * 120
    return train_byte_bpe_tokenizer([text], tmp_path / "tokenizer.json", vocab_size=vocab)


def test_capacity_predicts_the_z_score_it_will_actually_get(tmp_path):
    """The estimator has to be right, not merely pessimistic."""
    from geocentric.watermark import watermark_capacity

    torch.manual_seed(5)
    tokenizer = _tokenizer(tmp_path)
    m = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=512,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    )).eval()
    config = WatermarkConfig(identity="Predictable", delta=2.0)

    prompt = torch.tensor([tokenizer.encode("the keeper").ids])
    torch.manual_seed(9)
    out = m.generate(prompt, max_new_tokens=400, temperature=0.8, top_k=0, top_p=1.0,
                     repetition_penalty=1.0, logits_processor=WatermarkProcessor(
                         config, tokenizer.get_vocab_size()))
    ids = out[0].tolist()
    observed = detect_ids(ids, config, tokenizer.get_vocab_size())

    report = watermark_capacity(m, tokenizer, tokenizer.decode(ids), config, temperature=0.8)
    predicted_z = report.expected_z_per_sqrt_token * math.sqrt(observed.scored_tokens)

    assert observed.z_score > 4, "an untrained model has plenty of entropy; it must mark"
    # Within a factor of two of the realised z is a useful estimate; the point is that
    # it is on the right scale, not that it is exact.
    assert 0.5 < predicted_z / observed.z_score < 2.0, (predicted_z, observed.z_score)


def test_capacity_calls_a_near_deterministic_model_unwatermarkable(tmp_path):
    """The failure this exists to catch: a confident model carries no mark at all."""
    from geocentric.watermark import watermark_capacity

    tokenizer = _tokenizer(tmp_path)
    m = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=256,
        n_layer=1, n_head=2, n_kv_head=1, n_embd=32,
    )).eval()
    # Force near-total certainty by blowing up the logit scale.
    with torch.no_grad():
        m.lm_head.weight.mul_(400.0)

    config = WatermarkConfig(identity="Certain", delta=2.0)
    report = watermark_capacity(m, tokenizer, "the keeper walked along the harbour " * 20, config)
    assert report.median_entropy < 0.01
    assert report.expected_z_per_sqrt_token < 0.05
    assert "unwatermarkable" in report.verdict(config.gamma)


def test_capacity_rises_with_delta(tmp_path):
    from geocentric.watermark import watermark_capacity

    torch.manual_seed(2)
    tokenizer = _tokenizer(tmp_path)
    m = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=256,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    )).eval()
    text = "the keeper walked along the harbour and counted the boats " * 12
    weak = watermark_capacity(m, tokenizer, text, WatermarkConfig(identity="X", delta=0.5))
    strong = watermark_capacity(m, tokenizer, text, WatermarkConfig(identity="X", delta=6.0))
    assert strong.expected_z_per_sqrt_token > weak.expected_z_per_sqrt_token
    assert strong.tokens_for_decisive_z < weak.tokens_for_decisive_z


def test_capacity_handles_text_too_short_to_judge(tmp_path):
    from geocentric.watermark import watermark_capacity

    tokenizer = _tokenizer(tmp_path)
    m = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=64,
                                n_layer=1, n_head=2, n_kv_head=1, n_embd=32)).eval()
    report = watermark_capacity(m, tokenizer, "a", WatermarkConfig(identity="X"))
    assert report.scored_positions == 0
    assert "too little text" in report.verdict(0.25)
