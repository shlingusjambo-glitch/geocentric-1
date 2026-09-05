from __future__ import annotations

import math

import pytest
import torch

from geocentric.model import GPTConfig, GeocentricGPT, KVCache


def build(**kwargs) -> GeocentricGPT:
    config = GPTConfig(
        vocab_size=kwargs.pop("vocab_size", 512), block_size=kwargs.pop("block_size", 64),
        n_layer=kwargs.pop("n_layer", 2), n_head=kwargs.pop("n_head", 4),
        n_kv_head=kwargs.pop("n_kv_head", 2), n_embd=kwargs.pop("n_embd", 64), **kwargs,
    )
    return GeocentricGPT(config)


def test_initial_loss_is_near_uniform():
    model = build(vocab_size=512)
    ids = torch.randint(0, 512, (4, 32))
    labels = torch.randint(0, 512, (4, 32))
    _, loss = model(ids, labels=labels)
    assert abs(float(loss) - math.log(512)) < 0.6


def test_residual_projections_use_scaled_init():
    model = build(n_layer=8, n_embd=128, n_head=4, n_kv_head=2)
    expected = 0.02 / math.sqrt(2 * 8)
    assert abs(float(model.blocks[0].attn.proj.weight.std()) - expected) < expected * 0.25
    assert abs(float(model.blocks[0].mlp.down.weight.std()) - expected) < expected * 0.25


def test_kv_cache_matches_a_full_forward_pass():
    model = build()
    model.eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        full, _ = model(ids)
        caches = [KVCache() for _ in model.blocks]
        model(ids[:, :15], caches=caches, position_offset=0)
        cached, _ = model(ids[:, 15:16], caches=caches, position_offset=15)
    assert torch.allclose(full[0, -1], cached[0, -1], atol=1e-4)


def test_attention_is_causal():
    """A change to a later token must not alter an earlier position's logits."""
    model = build()
    model.eval()
    a = torch.randint(0, 512, (1, 16))
    b = a.clone()
    b[0, -1] = (int(b[0, -1]) + 7) % 512
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-5)


def test_embeddings_are_tied():
    model = build()
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_grouped_query_attention_shrinks_the_kv_projections():
    model = build(n_embd=128, n_head=8, n_kv_head=2)
    attn = model.blocks[0].attn
    assert attn.q_proj.weight.shape[0] == 128
    assert attn.k_proj.weight.shape[0] == 32
    assert attn.v_proj.weight.shape[0] == 32


def test_generate_respects_eos_and_context_limit():
    model = build(block_size=32)
    ids = torch.randint(0, 512, (1, 8))
    out = model.generate(ids, max_new_tokens=64, temperature=0.0)
    assert out.size(1) <= 32


def test_gradients_reach_every_parameter():
    model = build()
    ids = torch.randint(0, 512, (2, 16))
    _, loss = model(ids, labels=ids)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_config_round_trips(tmp_path):
    model = build(n_kv_head=2)
    model.config.save(tmp_path / "config.json")
    assert GPTConfig.load(tmp_path / "config.json") == model.config


def test_mismatched_head_counts_are_rejected():
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=100, n_embd=64, n_head=5)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=100, n_embd=64, n_head=4, n_kv_head=3)


def test_checkpoint_records_and_reports_its_stage(tmp_path):
    """chat picks its mode from this, so a wrong answer means the wrong prompt format."""
    from geocentric.checkpoint import checkpoint_stage, save_checkpoint

    model = build()
    save_checkpoint(model, tmp_path, step=1, name="m_pretrained.pt", extra={"stage": "pretrained"})
    assert checkpoint_stage(tmp_path) == "pretrained"

    save_checkpoint(model, tmp_path, step=2, name="m_sft.pt", extra={"stage": "sft"})
    # An SFT checkpoint outranks a pretrained one in the same directory.
    assert checkpoint_stage(tmp_path) == "sft"


def test_stage_falls_back_to_the_filename(tmp_path):
    """Checkpoints written before the stage field must still be classified."""
    import torch

    from geocentric.checkpoint import checkpoint_stage

    model = build()
    torch.save({"model": model.state_dict(), "config": vars(model.config), "step": 1},
               tmp_path / "legacy_sft.pt")
    assert checkpoint_stage(tmp_path) == "sft"


def test_missing_checkpoint_defaults_to_base_mode(tmp_path):
    """Guessing 'sft' for an absent checkpoint would apply a chat template blindly."""
    from geocentric.checkpoint import checkpoint_stage

    assert checkpoint_stage(tmp_path) == "pretrained"
