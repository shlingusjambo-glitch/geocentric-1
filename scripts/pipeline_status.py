"""Where the pipeline is, for anything that wants to show it.

The trainer's metrics file does not exist until training starts. Before that a
download runs for an hour, a tokenizer for another, and corpus tokenization for
several more -- and through all of it every monitor said "waiting", which looks
exactly like a run that has quietly died. This reports the earlier stages, and
the finished ones, from artefacts on disk.

stdlib only, and deliberately so: the web monitor imports this, and it must
stay unable to create a CUDA context or take a core from the trainer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STAGES = [
    (1, "download", "corpus and instruction data"),
    (2, "tokenizer", "byte-level BPE over this corpus"),
    (3, "tokenize corpus", "text to binary token shards"),
    (4, "pretrain", "from random initialisation"),
    (5, "instruction tuning", "chat, code, math, reasoning"),
]
TERMINAL = {"complete", "stopped", "failed", "diverged"}


_TARGETS_CACHE: dict | None = None


def _load_recipe_targets(default_tokens: float = 5e9, bytes_per_token: float = 4.2):
    """Per-source byte targets, read from the download recipe itself.

    Importing the recipe rather than restating its shares keeps one source of
    truth; it declares only constants at module level and pulls `datasets`
    inside its functions, so this stays stdlib-only.
    """
    global _TARGETS_CACHE
    if _TARGETS_CACHE is not None:
        return _TARGETS_CACHE
    recipe = Path(__file__).resolve().parent / "download_kestrel.py"
    if not recipe.exists():
        return {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kestrel_recipe", recipe)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        total = default_tokens * bytes_per_token
        _TARGETS_CACHE = {name: int(total * spec_["share"])
                          for name, spec_ in module.PRETRAIN_SOURCES.items()}
        return _TARGETS_CACHE
    except Exception:
        return {}


def read_pipeline(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def download_progress(data_dir: Path) -> dict:
    """Bytes on disk against the target for each pretraining source."""
    targets = _load_recipe_targets()
    pretrain = Path(data_dir) / "pretrain"
    sources = []
    done_bytes = target_bytes = 0
    for name, target in targets.items():
        path = pretrain / f"{name}.txt"
        try:
            have = path.stat().st_size
        except OSError:
            have = 0
        done_bytes += min(have, target)
        target_bytes += target
        sources.append({
            "name": name, "bytes": have, "target": target,
            "fraction": min(1.0, have / target) if target else 0.0,
            # A source is only complete once it stops growing at or past target;
            # the one currently being written is the one under target.
            "state": "done" if have >= target else ("running" if have else "pending"),
        })
    sft = Path(data_dir) / "sft" / "kestrel_sft.jsonl"
    try:
        sft_bytes = sft.stat().st_size
    except OSError:
        sft_bytes = 0
    return {
        "sources": sources,
        "bytes": done_bytes,
        "target": target_bytes,
        "fraction": (done_bytes / target_bytes) if target_bytes else 0.0,
        "sft_bytes": sft_bytes,
    }


def corpus_progress(run_dir: Path) -> dict:
    """Binary shard bytes, for the tokenize-corpus stage."""
    corpus = Path(run_dir) / "corpus"
    total = 0
    shards = 0
    for shard in corpus.glob("*.bin"):
        try:
            total += shard.stat().st_size
            shards += 1
        except OSError:
            pass
    return {"bytes": total, "shards": shards, "tokens": total // 2}


def describe(run_dir: str | Path, data_dir: str | Path) -> dict:
    """One dict covering every stage, whichever one is live."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    pipeline = read_pipeline(run_dir)
    stage = int(pipeline.get("stage") or 0)

    tokenizer = run_dir / "tokenizer.json"
    try:
        tokenizer_bytes = tokenizer.stat().st_size
    except OSError:
        tokenizer_bytes = 0

    metrics = {}
    try:
        metrics = json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    # Infer the stage when no recipe wrote one (a bare `pretrain` invocation, or
    # a run started before pipeline.json existed).
    if not stage:
        if metrics.get("phase") == "sft":
            stage = 5
        elif metrics:
            stage = 4
        elif corpus_progress(run_dir)["bytes"]:
            stage = 3
        elif tokenizer_bytes:
            stage = 2
        elif (Path(data_dir) / "pretrain").is_dir():
            stage = 1

    download = download_progress(data_dir)
    corpus = corpus_progress(run_dir)
    detail, fraction = "", 0.0
    if stage == 1:
        running = [s["name"] for s in download["sources"] if s["state"] == "running"]
        detail = (f"{running[0]} · " if running else "") + \
                 f"{download['bytes'] / 1e9:.1f} of {download['target'] / 1e9:.1f} GB"
        fraction = download["fraction"]
    elif stage == 2:
        detail = "training BPE" if not tokenizer_bytes else f"{tokenizer_bytes / 1e6:.1f} MB written"
        fraction = 1.0 if tokenizer_bytes else 0.0
    elif stage == 3:
        detail = f"{corpus['tokens'] / 1e9:.2f}B tokens in {corpus['shards']} shard(s)"
        fraction = 0.0
    elif stage in (4, 5):
        step = metrics.get("step") or 0
        total = (metrics.get("config") or {}).get("total_steps") or 0
        fraction = (step / total) if total else 0.0
        detail = f"step {step:,} of {total:,}" if total else "starting"

    name, blurb = "", ""
    for number, label, description in STAGES:
        if number == stage:
            name, blurb = label, description
    return {
        "stage": stage,
        "stages": len(STAGES),
        "stage_name": name,
        "stage_blurb": blurb,
        "stage_detail": detail,
        "stage_fraction": fraction,
        "stage_list": [{"n": n, "name": l,
                        "state": "done" if stage > n else ("live" if stage == n else "todo")}
                       for n, l, _ in STAGES],
        "started": pipeline.get("started"),
        "download": download,
        "tokenizer_bytes": tokenizer_bytes,
        "corpus": corpus,
    }
