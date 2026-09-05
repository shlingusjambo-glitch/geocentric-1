"""Live view of a training run. Reads the metrics file the trainer writes.

    python scripts/watch_training.py runs/geocentric-120m
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 48) -> str:
    if len(values) < 2:
        return ""
    if len(values) > width:
        # Average into buckets so the whole history stays visible.
        size = len(values) / width
        values = [
            sum(values[int(i * size): max(int((i + 1) * size), int(i * size) + 1)])
            / max(1, len(values[int(i * size): max(int((i + 1) * size), int(i * size) + 1)]))
            for i in range(width)
        ]
    low, high = min(values), max(values)
    span = high - low or 1.0
    return "".join(BLOCKS[min(len(BLOCKS) - 1, int((v - low) / span * (len(BLOCKS) - 1)))] for v in values)


def gpu_stats() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(", ")
        return f"{out[0]}% util | {int(out[1]):,}/{int(out[2]):,} MiB | {out[3]}°C | {float(out[4]):.0f}W"
    except Exception:
        return "unavailable"


def human_time(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "—"
    delta = timedelta(seconds=int(seconds))
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def bar(fraction: float, width: int = 40) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def render(run_dir: Path, history: list[float], eval_history: list[float]) -> str:
    metrics_path = run_dir / "training_metrics.json"
    corpus_tmp = run_dir / "corpus" / "corpus.tmp.bin"
    lines: list[str] = []
    width = min(shutil.get_terminal_size((100, 30)).columns, 100)

    lines.append("═" * width)
    lines.append(f"  GEOCENTRIC  ·  {run_dir}")
    lines.append("═" * width)

    if not metrics_path.exists():
        train_bin = run_dir / "corpus" / "train.bin"
        lines.append("")
        if train_bin.exists() and corpus_tmp.exists():
            # Both present means the split is running: the tokenized stream is being
            # copied into its train and validation halves.
            done = train_bin.stat().st_size
            whole = corpus_tmp.stat().st_size or 1
            lines.append("  Stage: writing train/validation split")
            lines.append(f"  {bar(done / whole, 40)}  {done / whole * 100:5.1f}%")
            lines.append(f"  {done / 1024**3:.2f} GB of {whole / 1024**3:.2f} GB copied")
            lines.append("")
            lines.append("  Training begins automatically when this finishes.")
        elif corpus_tmp.exists():
            size_gb = corpus_tmp.stat().st_size / 1024**3
            lines.append("  Stage: tokenizing corpus")
            lines.append(f"  Written: {size_gb:.2f} GB of token shards ({size_gb * 1024**3 / 2 / 1e9:.2f}B tokens)")
            lines.append("")
            lines.append("  Training begins automatically when this finishes.")
        else:
            lines.append("  Waiting for the run to start...")
        lines.append("")
        lines.append(f"  GPU: {gpu_stats()}")
        lines.append("═" * width)
        return "\n".join(lines)

    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "\n".join(lines + ["  (metrics file being written, retrying)"])

    cfg = m.get("config", {})
    step = m.get("step") or 0
    total = cfg.get("total_steps") or 1
    loss = m.get("loss")
    tps = m.get("tokens_per_second") or 0
    tok_per_step = cfg.get("tokens_per_step") or 0
    phase = m.get("phase", "?")
    status = m.get("status", "running")

    if loss is not None and (not history or history[-1] != loss):
        history.append(loss)
    ev = m.get("eval_loss")
    if ev is not None and (not eval_history or eval_history[-1] != ev):
        eval_history.append(ev)

    frac = step / total
    remaining = (total - step) * (tok_per_step / tps) if tps else 0

    lines.append("")
    lines.append(f"  Stage    {phase}  ·  {status}")
    lines.append(f"  Model    {cfg.get('params', 0):,} params · {cfg.get('n_layer','?')}L × {cfg.get('n_embd','?')}d "
                 f"· ctx {cfg.get('block_size','?')} · {str(cfg.get('dtype','')).replace('torch.','')}"
                 f"{' · compiled' if cfg.get('compiled') else ''}")
    lines.append("")
    lines.append(f"  {bar(frac, min(50, width - 30))}  {frac*100:5.1f}%")
    lines.append(f"  step {step:,} / {total:,}      ETA {human_time(remaining)}")
    lines.append("")

    if loss is not None:
        lines.append(f"  loss        {loss:.4f}      perplexity {math.exp(min(loss, 20)):,.1f}")
    if eval_history:
        lines.append(f"  eval loss   {eval_history[-1]:.4f}      best {min(eval_history):.4f}")
    lines.append(f"  throughput  {tps:,.0f} tok/s" + (f"      MFU {m['mfu']*100:.1f}%" if m.get("mfu") else ""))
    lines.append(f"  seen        {m.get('tokens_seen', 0):,} of {cfg.get('corpus_tokens', 0):,} tokens")
    lines.append(f"  lr          {m.get('lr', 0):.2e}")
    if m.get("peak_memory_gb"):
        lines.append(f"  peak VRAM   {m['peak_memory_gb']:.2f} GB")

    if len(history) > 2:
        lines.append("")
        lines.append(f"  train loss  {sparkline(history, min(48, width - 20))}")
        lines.append(f"              {max(history):.2f} → {history[-1]:.2f}")

    lines.append("")
    lines.append(f"  GPU  {gpu_stats()}")
    ckpts = sorted(run_dir.glob("*.pt"))
    if ckpts:
        newest = max(ckpts, key=lambda p: p.stat().st_mtime)
        age = time.time() - newest.stat().st_mtime
        lines.append(f"  last save  {newest.name} ({newest.stat().st_size/1024**3:.2f} GB, {human_time(age)} ago)")
    lines.append("")
    lines.append(f"  {datetime.now():%H:%M:%S}   Ctrl+C to stop watching (training keeps running)")
    lines.append("═" * width)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/geocentric-120m")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    history: list[float] = []
    eval_history: list[float] = []
    try:
        while True:
            frame = render(run_dir, history, eval_history)
            if args.once:
                print(frame)
                return
            print("\033[2J\033[H" + frame, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching. Training continues in the background.")


if __name__ == "__main__":
    main()
