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
    if hasattr(model, "_epicycle_state"):
        payload["epicycle_state"] = model._epicycle_state
    if hasattr(model, "_training_tokens"):
        payload["tokens_seen"] = model._training_tokens
    if getattr(model, "_sft_resume_state", None) is not None:
        payload["sft_resume_state"] = model._sft_resume_state
    if hasattr(model, "_grad_scaler"):
        payload["grad_scaler"] = model._grad_scaler.state_dict()
    # Optimizer state makes a resumed run continue with its real Adam moments and
    # schedule position instead of restarting the optimizer from zero, which
    # previously threw away progress on every interruption.
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["optimizer_type"] = type(optimizer).__name__
    if extra:
        payload.update(extra)

    path = out / name
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def resolve_model_target(model_dir: str | Path,
                         checkpoint_name: Optional[str] = None) -> Tuple[Path, Optional[str]]:
    """Split a target into (directory, checkpoint name), accepting either form.

    Pointing --model_dir straight at a .pt file is the natural thing to type
    when one run directory holds several checkpoints, so a file path means
    "this exact checkpoint" rather than a directory to search inside.
    """
    path = Path(model_dir)
    if path.suffix == ".pt":
        if not path.is_file():
            # Reporting "no checkpoint found in <file>" for a mistyped filename
            # sends the reader looking for a directory that was never meant.
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if checkpoint_name and checkpoint_name != path.name:
            raise ValueError(
                f"Conflicting checkpoints: --model_dir names {path.name} but "
                f"--checkpoint names {checkpoint_name}. Pass only one."
            )
        return path.parent, path.name
    return path, checkpoint_name


def _find_checkpoint(model_dir: Path, checkpoint_name: Optional[str]) -> Path:
    model_dir, checkpoint_name = resolve_model_target(model_dir, checkpoint_name)
    if checkpoint_name:
        candidate = model_dir / checkpoint_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Requested checkpoint not found: {candidate}")
    # Most-derived first: a vision checkpoint contains the SFT weights it was built on,
    # and an SFT checkpoint contains the pretrained ones.
    for pattern in ("*_vision_best.pt", "*_vision.pt", "*_sft_best.pt", "*_sft.pt",
                    "*_pretrained_best.pt", "*_pretrained.pt", "*.pt"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint (.pt) found in {model_dir}")


STAGES = ("vision", "sft", "pretrained")


def checkpoint_stage(model_dir: str | Path, checkpoint_name: Optional[str] = None) -> str:
    """Return "vision", "sft" or "pretrained" for the checkpoint that would be loaded.

    Prefers the stage recorded inside the checkpoint; falls back to the filename for
    checkpoints written before that field existed. "vision" implies instruction-tuned:
    vision training runs on chat-formatted image conversations, so those checkpoints
    take the chat path everywhere "sft" does.
    """
    try:
        path = _find_checkpoint(Path(model_dir), checkpoint_name)
    except FileNotFoundError:
        return "pretrained"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stage = payload.get("stage")
        if stage in STAGES:
            return stage
    except Exception:
        pass
    for stage in STAGES:
        if f"_{stage}" in path.name:
            return stage
    return "pretrained"


def load_checkpoint(
    model_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    checkpoint_name: Optional[str] = None,
    mmap: bool = False,
    **_ignored: Any,
) -> GeocentricGPT:
    path = _find_checkpoint(Path(model_dir), checkpoint_name)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)

    if "config" in payload:
        config = GPTConfig(**{k: v for k, v in payload["config"].items() if k in GPTConfig.__annotations__})
    else:
        config = GPTConfig.load(path.parent / "config.json")

    model = GeocentricGPT(config)
    if config.vision:
        # Build the tower first or every "vision.*" key lands in `unexpected` and the
        # architecture-mismatch guard below rejects a perfectly good checkpoint.
        from geocentric.vision import VisionConfig, VisionTower

        vision_config = VisionConfig.from_dict(config.vision)
        model.vision = VisionTower(vision_config, config.n_embd)
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
    model._epicycle_state = payload.get("epicycle_state")
    model._sft_optimizer_config = payload.get("sft_optimizer_config")
    model._sft_resume_state = payload.get("sft_resume_state")
    saved_optimizer_type = payload.get("optimizer_type")
    if saved_optimizer_type is None and "optimizer" in payload:
        states = payload["optimizer"].get("state", {}).values()
        saved_optimizer_type = "RingAdamW" if any("ring" in state for state in states) else "AdamW"
    model._checkpoint_optimizer_type = saved_optimizer_type
    model._training_tokens = payload.get("tokens_seen")
    model._grad_scaler_state = payload.get("grad_scaler")
    model._checkpoint_loss = payload.get("loss")
    model._checkpoint_step = int(payload.get("step", 0))
    print(f"Loaded checkpoint {path.name} (step {payload.get('step', 0):,})")
    if config.watermark:
        print(f"  watermarked as {config.watermark.get('identity')!r}")
    if config.vision:
        print(f"  multimodal: vision tower attached "
              f"({config.vision.get('image_size')}px, {config.vision.get('n_layer')} layers)")
    return model


def load_optimizer_state(model_dir: str | Path, checkpoint_name: Optional[str], optimizer, *, mmap=False) -> int:
    try:
        path = _find_checkpoint(Path(model_dir), checkpoint_name)
    except FileNotFoundError:
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    if "optimizer" in payload:
        saved_type = payload.get("optimizer_type")
        if saved_type is None:
            states = payload["optimizer"].get("state", {}).values()
            saved_type = "RingAdamW" if any("ring" in s for s in states) else "AdamW"
        if saved_type != type(optimizer).__name__:
            raise ValueError(f"Cannot resume {saved_type} state with {type(optimizer).__name__}; "
                             "use the original optimizer/EPICYCLE preset")
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception as exc:
            raise RuntimeError(f"Could not restore optimizer state: {exc}") from exc
    return int(payload.get("step", 0))


def find_tokenizer_path(model_dir: str | Path, extra_dirs: Iterable[str | Path] = ()) -> Path:
    model_dir, _ = resolve_model_target(model_dir)
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
