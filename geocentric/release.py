"""Package a trained checkpoint for other people to use.

`release` is the last point at which anyone will be asked about attribution, so it
is the point at which it is asked properly. A model that leaves this machine without
a watermark cannot be given one afterwards — the weights are out — which is why the
question is asked here even when it was already answered at training time.

Everything written is inspectable: weights, tokenizer, config, a model card, and the
PARALLAX report with the scores that card claims. No pickled setup code, no download
step.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from geocentric import __version__
from geocentric.checkpoint import _find_checkpoint, checkpoint_stage
from geocentric.watermark import WATERMARK_FILE, WatermarkConfig


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_watermark_to_checkpoint(
    checkpoint: Path, watermark: Optional[WatermarkConfig], remove: bool
) -> None:
    """Rewrite the identity inside the weights file itself.

    The config travels with the checkpoint, so the mark cannot be lost by someone
    copying the .pt and leaving watermark.json behind. Loading with
    `weights_only=False` is safe here: this is a file this process just wrote.
    """
    if watermark is None and not remove:
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        return
    config["watermark"] = watermark.to_dict() if watermark else None
    payload["config"] = config
    tmp = checkpoint.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(checkpoint)


def release(
    model_dir: str,
    output_dir: str,
    watermark: Optional[WatermarkConfig] = None,
    remove_watermark: bool = False,
    checkpoint_name: Optional[str] = None,
    license_name: str = "",
    description: str = "",
    benchmark: bool = True,
    eval_text: Optional[str] = None,
    vision_data: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    src = Path(model_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Model directory does not exist: {src}")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{out} already exists and is not empty. Pass --overwrite to replace it."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = _find_checkpoint(src, checkpoint_name)
    stage = checkpoint_stage(src, checkpoint_name)
    print(f"Releasing {checkpoint.name} ({stage}) -> {out}")

    # Weights, stripped of optimizer state: a release does not need to be resumable
    # and Adam moments are two thirds of the file.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in payload.items() if k != "optimizer"}
    released = out / checkpoint.name
    torch.save(slim, released)
    saved_mb = (checkpoint.stat().st_size - released.stat().st_size) / 1024**2
    print(f"  weights: {released.name} ({released.stat().st_size / 1024**2:,.0f} MB, "
          f"{saved_mb:,.0f} MB of optimizer state dropped)")

    # Three cases, and the middle one is the one that is easy to get wrong: an
    # unchanged watermark must be carried forward, not quietly dropped. It lives in
    # the checkpoint config, so read it from there rather than trusting the sidecar
    # to still be sitting next to the weights.
    inherited = WatermarkConfig.from_dict((slim.get("config") or {}).get("watermark"))
    effective = None if remove_watermark else (watermark or inherited)
    _apply_watermark_to_checkpoint(released, effective, remove_watermark)
    if effective is not None:
        effective.save(out)
        if watermark is None and inherited is not None:
            print(f"  watermark: carried forward as {effective.identity!r}")
    if remove_watermark:
        (out / WATERMARK_FILE).unlink(missing_ok=True)

    for name in ("tokenizer.json", "config.json", "vision.json", "vision_config.json"):
        source = src / name
        if source.exists():
            shutil.copyfile(source, out / name)
    # config.json is regenerated so it agrees with the watermark decision made above.
    config = dict(slim.get("config") or {})
    config["watermark"] = effective.to_dict() if effective else None
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report_path: Optional[Path] = None
    result = None
    if benchmark:
        from geocentric.parallax import run_parallax, write_report

        print("  running PARALLAX...")
        try:
            result = run_parallax(src, eval_text=eval_text, vision_data=vision_data)
            report_path = write_report(result, out / "PARALLAX.md")
            print(f"  benchmark: Parallax Index {result.index:.1f}/100 -> {report_path.name}")
        except Exception as exc:
            # A benchmark failure must not cost you the release; say so and continue.
            print(f"  benchmark skipped: {exc}")

    card = _model_card(
        config=config, stage=stage, watermark=effective,
        result=result, license_name=license_name, description=description,
        checkpoint=released, source=src,
    )
    (out / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    manifest: Dict[str, Any] = {
        "name": config.get("model_name", "Geocentric"),
        "geocentric_version": __version__,
        "stage": stage,
        "released": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "watermark": (effective.to_dict() if effective else None),
        "parallax_index": (result.index if result else None),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(out.iterdir()) if path.is_file()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nReleased to {out}")
    print(f"  {len(manifest['files']) + 1} files | "
          f"watermark: {effective.identity if effective else 'none'}")
    print(f"  Try it:  geocentric chat --model_dir {out}")
    return out


def _model_card(
    config: Dict[str, Any],
    stage: str,
    watermark: Optional[WatermarkConfig],
    result,
    license_name: str,
    description: str,
    checkpoint: Path,
    source: Path,
) -> str:
    name = config.get("model_name", "Geocentric")
    params = config.get("n_layer", 0), config.get("n_embd", 0)
    lines: List[str] = [
        f"# {name}",
        "",
        description or f"A {stage} causal language model trained from scratch with Geocentric "
                       f"{__version__}.",
        "",
        "## What it is",
        "",
        "| | |",
        "|---|---|",
        f"| Stage | {stage} |",
        f"| Layers x width | {params[0]} x {params[1]} |",
        f"| Attention heads | {config.get('n_head')} (kv {config.get('n_kv_head')}) |",
        f"| Context | {config.get('block_size'):,} tokens |" if config.get("block_size") else "| Context | — |",
        f"| Vocabulary | {config.get('vocab_size', 0):,} |",
        f"| Multimodal | {'yes — accepts images' if config.get('vision') else 'no, text only'} |",
        f"| Licence | {license_name or 'not specified'} |",
        "",
        "Architecture: pre-norm decoder blocks, RMSNorm, rotary position embeddings, "
        "grouped-query attention, SwiGLU feed-forward, tied embeddings.",
        "",
    ]

    lines += ["## Watermark", ""]
    if watermark:
        lines += [
            f"Output from this model is watermarked and identifies as **{watermark.identity}**.",
            "",
            "The watermark is a green-list bias on token selection: at each position the "
            "previous token seeds a split of the vocabulary, and tokens on the favoured side "
            "are slightly more likely to be chosen. A reader cannot see it. A one-sided z-test "
            "over roughly forty or more tokens recovers it, and only under this identity — "
            "testing the same text under any other name returns noise.",
            "",
            "```bash",
            f'geocentric detect --identity "{watermark.identity}" --model_dir . --text "<text>"',
            "```",
            "",
            "Limits, stated plainly: short passages cannot be decided, and rewriting the text "
            "removes the mark. It is attribution for unedited output, not a forensic guarantee.",
            "",
        ]
    else:
        lines += ["This model's output is **not** watermarked. Text it generates cannot be "
                  "attributed back to it by any mechanism shipped here.", ""]

    if result is not None:
        lines += [
            "## Measured quality",
            "",
            f"PARALLAX {result.version} scores this checkpoint at "
            f"**{result.index:.1f}/100** (Parallax Index). Full report in `PARALLAX.md`.",
            "",
            "| probe | score | measured |",
            "|---|---:|---|",
        ]
        for probe in result.probes:
            score = f"{probe.score:.1f}" if probe.score is not None else "—"
            lines.append(f"| {probe.name} | {score} | {probe.headline} |")
        lines.append("")
        weak = sorted([p for p in result.probes if p.ran], key=lambda p: p.score or 0)[:2]
        if weak:
            lines += ["**Known weaknesses.** " + " ".join(
                f"{p.name} scores {p.score:.0f}/100 — {p.note}." for p in weak
            ), ""]

    lines += [
        "## Using it",
        "",
        "```bash",
        "pip install geocentric",
        "geocentric chat --model_dir .",
        "```",
        "",
        "```python",
        "from geocentric.checkpoint import load_model_and_tokenizer",
        "from geocentric.generate import generate_text",
        "",
        'model, tokenizer = load_model_and_tokenizer(".")',
        'print(generate_text(model, tokenizer, "The capital of France is"))',
        "```",
        "",
        "The watermark, if present, is applied automatically by `generate_text` and "
        "`stream_text` because the identity travels inside the checkpoint config.",
        "",
        "## Intended use and limits",
        "",
        "This is a small model trained from scratch. It will state false things with "
        "confidence, and the smaller the parameter count and the further below the "
        "compute-optimal token budget it was trained, the more often. Do not use it as a "
        "source of facts, for anything safety-critical, or unsupervised in front of users. "
        "It reflects whatever was in its training corpus, including that corpus's biases.",
        "",
        f"Released {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from `{source.name}` "
        f"({checkpoint.name}).",
        "",
    ]
    return "\n".join(lines)
