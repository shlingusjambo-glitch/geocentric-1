from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def _metrics_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "training_metrics.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _write_metrics(path: Path, payload: dict) -> None:
    # A watcher or interrupted write must never see half a JSON document.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize_training_metrics(output_dir: str | Path, phase: str, config: dict[str, Any]) -> None:
    path = _metrics_path(output_dir)
    payload = {
        "phase": phase,
        "status": "running",
        "config": config,
        "epoch": 0,
        "epochs": config.get("epochs", 0),
        "step": 0,
        "batch": 0,
        "loss": None,
        "perplexity": None,
        "eval_loss": None,
        "best_eval_loss": None,
        "history": {
            "steps": [],
            "loss": [],
            "eval_loss": [],
            "eval_steps": [],
            "eval_elapsed": [],
            # Wall clock alongside the loss, so "which run learns faster" can be
            # answered in seconds rather than in optimizer steps. A technique that
            # trades steps for speed is invisible on a step axis.
            "elapsed": [],
        },
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "use_scaler": config.get("use_scaler", False),
        "start_time": _now_iso(),
        "last_update": _now_iso(),
        "message": "Training started.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_metrics(path, payload)


def update_training_metrics(output_dir: str | Path, updates: dict[str, Any]) -> None:
    path = _metrics_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Training metrics not initialized: {path}")
    metrics = json.loads(path.read_text())
    # Append to history when loss/step present
    if "loss" in updates and updates.get("loss") is not None:
        try:
            history = metrics.setdefault("history", {})
            history.setdefault("steps", []).append(int(updates.get("step", metrics.get("step", 0))))
            history.setdefault("loss", []).append(float(updates.get("loss")))
            history.setdefault("elapsed", []).append(round(_elapsed_seconds(metrics), 2))
        except Exception:
            pass
    if "eval_loss" in updates and updates.get("eval_loss") is not None:
        try:
            metrics.setdefault("history", {}).setdefault("eval_loss", []).append(float(updates.get("eval_loss")))
            metrics["history"].setdefault("eval_steps", []).append(int(updates.get("step", metrics.get("step", 0))))
            metrics["history"].setdefault("eval_elapsed", []).append(round(_elapsed_seconds(metrics), 2))
        except Exception:
            pass
    metrics.update(updates)
    metrics["last_update"] = _now_iso()
    _write_metrics(path, metrics)


def _elapsed_seconds(metrics: dict[str, Any]) -> float:
    started = metrics.get("start_time")
    if not started:
        return 0.0
    try:
        start = datetime.fromisoformat(str(started).rstrip("Z"))
    except ValueError:
        return 0.0
    return (datetime.utcnow() - start).total_seconds()


def load_training_metrics(output_dir: str | Path) -> dict[str, Any]:
    path = _metrics_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Training metrics file not found: {path}")
    return json.loads(path.read_text())
