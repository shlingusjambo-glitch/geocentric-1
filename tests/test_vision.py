from __future__ import annotations

import json

import pytest
import torch

from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.tokenizer_train import train_byte_bpe_tokenizer
from geocentric.vision import (
    IMAGE_TOKEN,
    VisionConfig,
    VisionTower,
    attach_vision,
    image_placeholder,
    load_image,
)

pytest.importorskip("PIL")


def tiny_vision_config(**kwargs) -> VisionConfig:
    base = dict(image_size=64, patch_size=16, n_embd=64, n_layer=2, n_head=4, pool=2)
    base.update(kwargs)
    return VisionConfig(**base)


@pytest.fixture
def tokenizer(tmp_path):
    text = "the quick brown fox jumps over the lazy dog near the harbour wall " * 200
    return train_byte_bpe_tokenizer([text], tmp_path / "tokenizer.json", vocab_size=500)


@pytest.fixture
def lm(tokenizer):
    return GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=256,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    ))


def test_pooling_reduces_the_image_token_budget():
    unpooled = tiny_vision_config(pool=1)
    pooled = tiny_vision_config(pool=2)
    assert unpooled.n_tokens == 16
    assert pooled.n_tokens == 4
    # 224px at patch 16 is the realistic case: 196 tokens is a fifth of a 1024 context.
    real = VisionConfig()
    assert real.n_patches == 196 and real.n_tokens == 49


def test_bad_geometry_is_rejected_up_front():
    with pytest.raises(ValueError):
        VisionConfig(image_size=100, patch_size=16).grid
    with pytest.raises(ValueError):
        VisionConfig(image_size=64, patch_size=16, pool=3).n_tokens


def test_tower_encodes_to_the_language_models_width():
    config = tiny_vision_config()
    tower = VisionTower(config, lm_embd=128)
    out = tower.encode(torch.randn(3, 3, 64, 64))
    assert out.shape == (3, config.n_tokens, 128)


def test_the_vision_tower_is_not_causal():
    """A patch must be able to see the whole picture, not only what precedes it."""
    config = tiny_vision_config(pool=1)
    tower = VisionTower(config, lm_embd=64).eval()
    a = torch.randn(1, 3, 64, 64)
    b = a.clone()
    b[:, :, -16:, -16:] = torch.randn(1, 3, 16, 16)  # change the last patch only
    with torch.no_grad():
        out_a, out_b = tower.encode(a), tower.encode(b)
    assert not torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-5), \
        "the first patch did not react to a change in the last one — attention is masked"


def test_attach_costs_nothing_on_a_current_tokenizer(tokenizer, lm):
    """<|image|> is reserved at tokenizer-training time, so the usual path grows nothing."""
    assert tokenizer.token_to_id(IMAGE_TOKEN) is not None
    before = lm.config.vocab_size
    attach_vision(lm, tokenizer, tiny_vision_config())
    assert lm.config.vocab_size == before
    assert VisionConfig.from_dict(lm.config.vision).image_token_id == tokenizer.token_to_id(IMAGE_TOKEN)


def test_attach_grows_the_embedding_for_a_tokenizer_without_the_token(tmp_path):
    """A checkpoint from before vision existed must not need a retrain."""
    import geocentric.tokenizer_train as tt

    original = tt.SPECIAL_TOKENS
    tt.SPECIAL_TOKENS = [t for t in original if t != IMAGE_TOKEN]
    try:
        legacy = tt.train_byte_bpe_tokenizer(
            ["the quick brown fox jumps over the lazy dog " * 200],
            tmp_path / "legacy.json", vocab_size=400,
        )
    finally:
        tt.SPECIAL_TOKENS = original
    assert legacy.token_to_id(IMAGE_TOKEN) is None

    model = GeocentricGPT(GPTConfig(
        vocab_size=legacy.get_vocab_size(), block_size=128,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    ))
    before = model.config.vocab_size
    attach_vision(model, legacy, tiny_vision_config())

    assert legacy.token_to_id(IMAGE_TOKEN) is not None
    assert model.config.vocab_size > before
    assert model.token_embedding.num_embeddings == model.config.vocab_size
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr(), "still tied"
    assert torch.isfinite(model.token_embedding.weight).all()
    # The new row is seeded with the mean of the existing ones, not left at zero or noise.
    new_row = model.token_embedding.weight[before:]
    assert float(new_row.abs().max()) < float(model.token_embedding.weight[:before].abs().max())


def test_splice_replaces_placeholders_and_changes_the_output(tokenizer, lm):
    attach_vision(lm, tokenizer, tiny_vision_config())
    config = VisionConfig.from_dict(lm.config.vision)
    ids = torch.tensor([tokenizer.encode(image_placeholder(config) + "\na dog").ids])
    assert int((ids == config.image_token_id).sum()) == config.n_tokens

    lm.eval()
    with torch.no_grad():
        without, _ = lm(ids)
        first, _ = lm(ids, images=torch.randn(1, 3, 64, 64))
        second, _ = lm(ids, images=torch.randn(1, 3, 64, 64))
    assert not torch.allclose(without, first, atol=1e-5)
    assert not torch.allclose(first, second, atol=1e-5), "output must depend on the pixels"


def test_placeholder_count_mismatch_is_a_clear_error(tokenizer, lm):
    attach_vision(lm, tokenizer, tiny_vision_config())
    config = VisionConfig.from_dict(lm.config.vision)
    ids = torch.tensor([tokenizer.encode(image_placeholder(config)).ids])
    with pytest.raises(ValueError, match="placeholders"):
        lm(ids, images=torch.randn(2, 3, 64, 64))


def test_images_without_a_tower_is_a_clear_error(lm):
    with pytest.raises(ValueError, match="no vision tower"):
        lm(torch.zeros(1, 4, dtype=torch.long), images=torch.randn(1, 3, 64, 64))


def test_a_vision_checkpoint_round_trips(tmp_path, tokenizer, lm):
    from geocentric.checkpoint import load_checkpoint, save_checkpoint

    attach_vision(lm, tokenizer, tiny_vision_config())
    config = VisionConfig.from_dict(lm.config.vision)
    lm.eval()
    ids = torch.tensor([tokenizer.encode(image_placeholder(config) + "\nx").ids])
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected, _ = lm(ids, images=images)

    save_checkpoint(lm, tmp_path, 1, name="m_vision.pt", extra={"stage": "vision"})
    restored = load_checkpoint(tmp_path, device=torch.device("cpu"))
    assert restored.vision is not None
    with torch.no_grad():
        actual, _ = restored(ids, images=images)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_load_image_centre_crops_and_normalizes(tmp_path):
    from PIL import Image

    Image.new("RGB", (200, 100), (255, 0, 0)).save(tmp_path / "wide.png")
    tensor = load_image(tmp_path / "wide.png", size=32)
    assert tensor.shape == (3, 32, 32)
    # Pure red at mean 0.5 / std 0.5 maps to +1 on the red channel and -1 elsewhere.
    assert abs(float(tensor[0].mean()) - 1.0) < 1e-3
    assert abs(float(tensor[1].mean()) + 1.0) < 1e-3


def test_vision_dataset_builds_masked_examples(tmp_path, tokenizer, lm):
    from PIL import Image

    from geocentric.train_vision import VisionCollate, VisionSFTDataset

    attach_vision(lm, tokenizer, tiny_vision_config())
    config = VisionConfig.from_dict(lm.config.vision)
    for name, colour in (("a.png", (255, 0, 0)), ("b.png", (0, 0, 255))):
        Image.new("RGB", (64, 64), colour).save(tmp_path / name)
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps({"image": "a.png", "caption": "a red square"}) + "\n"
        + json.dumps({"image": "b.png", "caption": "a blue square"}) + "\n",
        encoding="utf-8",
    )

    dataset = VisionSFTDataset(tokenizer, tmp_path / "pairs.jsonl", config, block_size=256)
    assert len(dataset) == 2
    item = dataset[0]
    assert item["images"].shape == (1, 3, 64, 64)
    # Only the assistant's caption is supervised.
    assert int((item["labels"] != -100).sum()) > 0
    assert int((item["labels"] != -100).sum()) < item["labels"].numel()

    batch = VisionCollate(pad_id=0)([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 3, 64, 64)
    assert int((batch["input_ids"] == config.image_token_id).sum()) == 2 * config.n_tokens
    _, loss = lm(batch["input_ids"], labels=batch["labels"], images=batch["images"])
    assert torch.isfinite(loss)


def test_missing_images_are_reported_not_silently_dropped(tmp_path, tokenizer, lm):
    from geocentric.train_vision import VisionSFTDataset

    attach_vision(lm, tokenizer, tiny_vision_config())
    config = VisionConfig.from_dict(lm.config.vision)
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps({"image": "gone.png", "caption": "nothing"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing or unreadable"):
        VisionSFTDataset(tokenizer, tmp_path / "pairs.jsonl", config, block_size=256)


def test_a_vision_checkpoint_is_recognised_as_instruction_tuned(tmp_path, tokenizer, lm):
    """Vision training runs on chat-formatted data, so it must take the chat path."""
    from geocentric.checkpoint import checkpoint_stage, save_checkpoint

    attach_vision(lm, tokenizer, tiny_vision_config())
    save_checkpoint(lm, tmp_path, 1, name="m_pretrained.pt", extra={"stage": "pretrained"})
    assert checkpoint_stage(tmp_path) == "pretrained"

    save_checkpoint(lm, tmp_path, 2, name="m_vision.pt", extra={"stage": "vision"})
    # Most-derived checkpoint wins, and its stage is reported honestly.
    assert checkpoint_stage(tmp_path) == "vision"


def test_stage_falls_back_to_the_filename_for_an_old_checkpoint(tmp_path):
    from geocentric.checkpoint import checkpoint_stage

    model = GeocentricGPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2,
                                    n_kv_head=1, n_embd=32))
    torch.save({"model": model.state_dict(), "step": 1}, tmp_path / "m_sft.pt")
    assert checkpoint_stage(tmp_path) == "sft"
