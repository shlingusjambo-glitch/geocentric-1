from __future__ import annotations

import json

import pytest
import torch

from geocentric.checkpoint import save_checkpoint
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.release import release
from geocentric.tokenizer_train import train_byte_bpe_tokenizer
from geocentric.watermark import WatermarkConfig


@pytest.fixture
def run_dir(tmp_path):
    """A finished run: weights with optimizer state, a tokenizer, a watermark."""
    source = tmp_path / "run"
    source.mkdir()
    tokenizer = train_byte_bpe_tokenizer(
        ["the keeper counted the boats in the harbour " * 100],
        source / "tokenizer.json", vocab_size=400,
    )
    model = GeocentricGPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), block_size=128,
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
        watermark=WatermarkConfig(identity="Original Author").to_dict(),
    ))
    vocab = tokenizer.get_vocab_size()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, vocab, (1, 16))
    model(ids, labels=ids)[1].backward()
    optimizer.step()
    save_checkpoint(model, source, 500, name="m_sft.pt", optimizer=optimizer,
                    extra={"stage": "sft"})
    WatermarkConfig(identity="Original Author").save(source)
    return source


def test_release_writes_a_complete_package(run_dir, tmp_path):
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False,
                  license_name="apache-2.0")
    names = {p.name for p in out.iterdir()}
    assert {"m_sft.pt", "tokenizer.json", "config.json", "MODEL_CARD.md",
            "manifest.json", "watermark.json"} <= names
    assert "apache-2.0" in (out / "MODEL_CARD.md").read_text()


def test_release_strips_optimizer_state(run_dir, tmp_path):
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    payload = torch.load(out / "m_sft.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert "model" in payload
    assert (out / "m_sft.pt").stat().st_size < (run_dir / "m_sft.pt").stat().st_size


def test_release_does_not_package_a_vision_dataset(run_dir, tmp_path):
    (run_dir / "vision.json").write_text('[{"image":"private.png","caption":"training only"}]')
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    assert not (out / "vision.json").exists()


def test_release_can_change_the_identity(run_dir, tmp_path):
    new = WatermarkConfig(identity="Renamed At Release", delta=4.0)
    out = release(str(run_dir), str(tmp_path / "dist"), watermark=new, benchmark=False)

    # Everywhere it could be read from must agree.
    assert WatermarkConfig.load(out).identity == "Renamed At Release"
    assert json.loads((out / "config.json").read_text())["watermark"]["identity"] == "Renamed At Release"
    payload = torch.load(out / "m_sft.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["watermark"]["identity"] == "Renamed At Release"
    assert payload["config"]["watermark"]["delta"] == 4.0


def test_release_can_remove_the_watermark(run_dir, tmp_path):
    out = release(str(run_dir), str(tmp_path / "dist"), watermark=None,
                  remove_watermark=True, benchmark=False)
    assert not (out / "watermark.json").exists()
    assert json.loads((out / "config.json").read_text())["watermark"] is None
    payload = torch.load(out / "m_sft.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["watermark"] is None
    assert "is **not** watermarked" in (out / "MODEL_CARD.md").read_text()


def test_a_released_model_still_marks_its_own_output(run_dir, tmp_path):
    """The identity has to survive inside the weights, not only in a sidecar file."""
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.generate import make_processor

    out = release(str(run_dir), str(tmp_path / "dist"),
                  watermark=WatermarkConfig(identity="Shipped"), benchmark=False)
    (out / "watermark.json").unlink()  # simulate someone copying only the .pt
    model, _ = load_model_and_tokenizer(out)
    processor = make_processor(model)
    assert processor is not None and processor.config.identity == "Shipped"


def test_the_manifest_hashes_every_file(run_dir, tmp_path):
    import hashlib

    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    manifest = json.loads((out / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
        assert entry["bytes"] == (out / name).stat().st_size


def test_release_refuses_to_clobber_without_overwrite(run_dir, tmp_path):
    release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    with pytest.raises(FileExistsError, match="overwrite"):
        release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    release(str(run_dir), str(tmp_path / "dist"), benchmark=False, overwrite=True)


def test_release_survives_a_failing_benchmark(run_dir, tmp_path, monkeypatch):
    """Losing the release because the benchmark crashed would be the wrong trade."""
    import geocentric.parallax as parallax

    def boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(parallax, "run_parallax", boom)
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=True)
    assert (out / "m_sft.pt").exists()
    assert not (out / "PARALLAX.md").exists()
    # And the card makes no quality claim it cannot back.
    assert "PARALLAX" not in (out / "MODEL_CARD.md").read_text()


def test_an_unchanged_watermark_is_carried_forward_not_dropped(run_dir, tmp_path):
    """Releasing without mentioning the watermark must not quietly remove it."""
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    assert WatermarkConfig.load(out).identity == "Original Author"
    assert json.loads((out / "config.json").read_text())["watermark"]["identity"] == "Original Author"
    payload = torch.load(out / "m_sft.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["watermark"]["identity"] == "Original Author"
    assert "Original Author" in (out / "MODEL_CARD.md").read_text()
    assert json.loads((out / "manifest.json").read_text())["watermark"]["identity"] == "Original Author"


def test_carrying_forward_works_from_the_checkpoint_alone(run_dir, tmp_path):
    """The sidecar can be missing; the identity in the weights is the source of truth."""
    (run_dir / "watermark.json").unlink()
    out = release(str(run_dir), str(tmp_path / "dist"), benchmark=False)
    assert WatermarkConfig.load(out).identity == "Original Author"
