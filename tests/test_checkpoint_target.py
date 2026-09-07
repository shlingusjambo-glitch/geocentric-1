"""--model_dir accepts a run directory or one checkpoint file inside it.

A run directory holds several checkpoints at once (pretrained, sft, best of
each), so naming the exact .pt file is the natural way to ask for one of them.
Directory targets keep their precedence search; a file target overrides it.
"""
from __future__ import annotations

import pytest
import torch

from geocentric.checkpoint import (
    checkpoint_stage,
    find_tokenizer_path,
    load_checkpoint,
    pretrained_checkpoint_name,
    resolve_model_target,
    save_checkpoint,
    sft_checkpoint_name,
)
from geocentric.model import GPTConfig, GeocentricGPT


def tiny_model():
    return GeocentricGPT(GPTConfig(
        vocab_size=256, block_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
    ))


@pytest.fixture
def run_dir(tmp_path):
    """A run holding both a pretrained and a (more derived) SFT checkpoint."""
    model = tiny_model()
    save_checkpoint(model, tmp_path, 100, name=pretrained_checkpoint_name("m"),
                    extra={"stage": "pretrained"})
    save_checkpoint(model, tmp_path, 200, name=sft_checkpoint_name("m"),
                    extra={"stage": "sft"})
    return tmp_path


def test_a_directory_target_is_unchanged(run_dir):
    assert resolve_model_target(run_dir) == (run_dir, None)


def test_a_file_target_splits_into_directory_and_name(run_dir):
    target = run_dir / pretrained_checkpoint_name("m")
    assert resolve_model_target(target) == (run_dir, target.name)


def test_a_file_target_overrides_the_precedence_search(run_dir):
    """The directory search prefers SFT; naming the pretrained file must win."""
    pretrained = run_dir / pretrained_checkpoint_name("m")

    from_directory = load_checkpoint(run_dir, device=torch.device("cpu"))
    from_file = load_checkpoint(pretrained, device=torch.device("cpu"))

    assert from_directory._checkpoint_step == 200
    assert from_file._checkpoint_step == 100


def test_the_stage_follows_the_named_file(run_dir):
    assert checkpoint_stage(run_dir) == "sft"
    assert checkpoint_stage(run_dir / pretrained_checkpoint_name("m")) == "pretrained"


def test_the_tokenizer_resolves_from_a_file_target(run_dir):
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    found = find_tokenizer_path(run_dir / sft_checkpoint_name("m"))
    assert found == run_dir / "tokenizer.json"


def test_config_json_still_backs_a_payload_without_a_config(run_dir):
    """The config fallback must look beside the checkpoint, not inside it."""
    model = tiny_model()
    path = run_dir / "bare.pt"
    torch.save({"model": model.state_dict(), "step": 7}, path)
    model.config.save(run_dir / "config.json")

    restored = load_checkpoint(path, device=torch.device("cpu"))
    assert restored.config.n_layer == model.config.n_layer


def test_a_redundant_matching_checkpoint_flag_is_accepted(run_dir):
    name = sft_checkpoint_name("m")
    assert resolve_model_target(run_dir / name, name) == (run_dir, name)


def test_two_different_checkpoints_are_rejected(run_dir):
    with pytest.raises(ValueError, match="Conflicting checkpoints"):
        resolve_model_target(run_dir / sft_checkpoint_name("m"),
                             pretrained_checkpoint_name("m"))


def test_a_mistyped_checkpoint_names_the_file_it_could_not_find(run_dir):
    """Not "no checkpoint found in <file>", which sends the reader hunting a directory."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_model_target(run_dir / "typo.pt")
