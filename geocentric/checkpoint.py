from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.tokenizer_train import load_tokenizer


def modelver_to_filename(modelver: str | None = None) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (modelver or "geocentric").strip().lower()).strip("_")
    return slug or "geocentric"


def pretrained_checkpoint_name(modelver: str | None = None, *, best: bool = False) -> str:
    return f"{modelver_to_filename(modelver)}_pretrained{'_best' if best else ''}.pt"


def sft_checkpoint_name(modelver: str | None = None, *, best: bool = False) -> str:
    return f"{modelver_to_filename(modelver)}_sft{'_best' if best else ''}.pt"


def save_checkpoint(
    model: GeocentricGPT,
    output_dir: str | Path,
    step: int,
    name: str = "model.pt",
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save(out / "config.json")

    payload: Dict[str, Any] = {
        "model": {k: v for k, v in model.state_dict().items()},
        "config": vars(model.config).copy(),
        "step": step,
    }
    # Optimizer state makes a resumed run continue with its real Adam moments and
    # schedule position instead of restarting the optimizer from zero, which
    # previously threw away progress on every interruption.
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)

    path = out / name
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def _find_checkpoint(model_dir: Path, checkpoint_name: Optional[str]) -> Path:
    if checkpoint_name:
        candidate = model_dir / checkpoint_name
        if candidate.exists():
            return candidate
    # Prefer a finished SFT model, then the best pretrained, then anything else.
    for pattern in ("*_sft_best.pt", "*_sft.pt", "*_pretrained_best.pt", "*_pretrained.pt", "*.pt"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint (.pt) found in {model_dir}")


def checkpoint_stage(model_dir: str | Path, checkpoint_name: Optional[str] = None) -> str:
    """Return "sft" or "pretrained" for the checkpoint that would be loaded.

    Prefers the stage recorded inside the checkpoint; falls back to the filename
    for checkpoints written before that field existed.
    """
    try:
        path = _find_checkpoint(Path(model_dir), checkpoint_name)
    except FileNotFoundError:
        return "pretrained"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stage = payload.get("stage")
        if stage in {"sft", "pretrained"}:
            return stage
    except Exception:
        pass
    return "sft" if "_sft" in path.name else "pretrained"


def load_checkpoint(
    model_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    checkpoint_name: Optional[str] = None,
    **_ignored: Any,
) -> GeocentricGPT:
    path = _find_checkpoint(Path(model_dir), checkpoint_name)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "config" in payload:
        config = GPTConfig(**{k: v for k, v in payload["config"].items() if k in GPTConfig.__annotations__})
    else:
        config = GPTConfig.load(Path(model_dir) / "config.json")

    model = GeocentricGPT(config)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {path.name} does not match the current architecture.\n"
            f"  missing: {sorted(missing)[:6]}\n  unexpected: {sorted(unexpected)[:6]}\n"
            "Checkpoints from Geocentric 2.1 and earlier used learned position embeddings, "
            "LayerNorm and a GELU MLP; they cannot be loaded into the current model. Retrain from scratch."
        )
    model = model.to(device=device, dtype=dtype if dtype != torch.float16 else torch.float32)
    print(f"Loaded checkpoint {path.name} (step {payload.get('step', 0):,})")
    return model


def load_optimizer_state(model_dir: str | Path, checkpoint_name: Optional[str], optimizer) -> int:
    try:
        path = _find_checkpoint(Path(model_dir), checkpoint_name)
    except FileNotFoundError:
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception as exc:
            print(f"Could not restore optimizer state ({exc}); continuing with a fresh optimizer.")
    return int(payload.get("step", 0))


def find_tokenizer_path(model_dir: str | Path, extra_dirs: Iterable[str | Path] = ()) -> Path:
    roots = [Path(model_dir), *[Path(d) for d in extra_dirs], Path.cwd()]
    for root in roots:
        candidate = Path(root) / "tokenizer.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"tokenizer.json not found near {model_dir}")


def load_model_and_tokenizer(
    model_dir: str | Path,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    checkpoint_name: Optional[str] = None,
    with_stage: bool = False,
):
    from geocentric.device import select_device

    device = device or select_device()
    model = load_checkpoint(model_dir, device=device, dtype=dtype, checkpoint_name=checkpoint_name)
    tokenizer = load_tokenizer(find_tokenizer_path(model_dir))
    model.eval()
    if with_stage:
        return model, tokenizer, checkpoint_stage(model_dir, checkpoint_name)
    return model, tokenizer
