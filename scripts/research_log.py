"""Autonomous research logger: benchmark the model at milestones, never disturb training.

Runs as a background daemon alongside a training run. At each milestone step it
loads the newest checkpoint **on CPU**, runs a fixed benchmark suite, and writes a
dated markdown report. Training is never signalled, paused or stopped — the GPU is
not touched, so the run continues at full speed throughout.

    nohup .venv/bin/python scripts/research_log.py runs/geocentric-120m &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Dense early where capability changes fastest, sparser later where it plateaus.
MILESTONES = [750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 5000,
              6000, 7000, 8000, 9000, 10000, 10588]

# Fixed suite, identical at every milestone, so differences are the model changing
# and nothing else. Each probes a capability that emerges at a different stage.
PROMPTS = [
    ("factual",       "The capital of France is"),
    ("science",       "Photosynthesis is the process by which"),
    ("definition",    "A large language model is"),
    ("narrative",     "The old lighthouse keeper walked down to the shore and"),
    ("instructional", "To bake bread, first"),
    ("list",          "Three reasons to exercise regularly:"),
    ("code",          "```python\ndef fibonacci(n):"),
    ("dialogue",      '"Where are you going?" she asked. He'),
    ("arithmetic",    "If a train leaves at 3pm and travels for two hours, it arrives at"),
    ("conversational", "Hello! What can you help me with?"),
]

SEED = 1234  # fixed so sampling noise does not masquerade as progress


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def read_metrics(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def trainer_alive(run_dir: Path) -> bool:
    pid_file = run_dir / "trainer.pid"
    pids = []
    if pid_file.exists():
        try:
            pids.append(int(pid_file.read_text().strip()))
        except (ValueError, OSError):
            pass
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except (OSError, PermissionError):
            continue
        if "geocentric.cli" in argv and "pretrain" in argv:
            return True
    return any(Path(f"/proc/{p}").exists() for p in pids)


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        return f"{out[0]}% util, {int(out[1]):,} MiB, {out[2]}°C, {float(out[3]):.0f} W"
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Benchmarks — all on CPU, so the training GPU is untouched
# ---------------------------------------------------------------------------

def run_benchmark(run_dir: Path, heldout_text: str, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)  # leave cores for the training dataloader
    from geocentric.checkpoint import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(run_dir, device=torch.device("cpu"))
    model.eval()
    result: dict = {"params": model.num_params(), "block_size": model.config.block_size}

    # --- held-out likelihood, tokenizer-independent and comparable over time ---
    @torch.no_grad()
    def score(text: str, max_windows: int = 10) -> tuple[float, float]:
        block = model.config.block_size
        ids = tok.encode(text).ids
        total_nll, counted, windows = 0.0, 0, 0
        for start in range(0, len(ids) - block - 1, block):
            if windows >= max_windows:
                break
            x = torch.tensor([ids[start:start + block]])
            y = torch.tensor([ids[start + 1:start + block + 1]])
            _, loss = model(x, labels=y)
            total_nll += float(loss) * y.numel()
            counted += y.numel()
            windows += 1
        if not counted:
            return float("nan"), float("nan")
        nats = total_nll / counted
        chars = len(text[:counted * 5].encode("utf-8")) or 1
        return nats, (total_nll / math.log(2)) / chars

    nats, bpb = score(heldout_text)
    result["heldout_nats_per_token"] = nats
    result["heldout_perplexity"] = math.exp(min(nats, 20))
    result["heldout_bits_per_byte"] = bpb

    # --- the code/prose split: code is lower-entropy, and the gap tracks learning ---
    try:
        raw = open(REPO / "data/pretrain/cosmopedia.txt", encoding="utf-8").read(60_000_000)
        docs = raw.split("\n\n\n")
        code = "\n\n".join([d for d in docs if "```python" in d and 1500 < len(d) < 6000][:12])
        prose = "\n\n".join([d for d in docs if "```" not in d and 1500 < len(d) < 6000][:12])
        result["code_nats"], _ = score(code, max_windows=6)
        result["prose_nats"], _ = score(prose, max_windows=6)
    except Exception as exc:
        result["code_nats"] = result["prose_nats"] = float("nan")
        log(f"code/prose split skipped: {exc}")

    # --- fixed generations ---
    generations = []
    for name, prompt in PROMPTS:
        torch.manual_seed(SEED)  # identical sampling stream at every milestone
        ids = torch.tensor([tok.encode(prompt).ids])
        started = time.perf_counter()
        out = model.generate(ids, max_new_tokens=80, temperature=0.7, top_k=50,
                             top_p=0.95, repetition_penalty=1.1)
        text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
        generations.append({"name": name, "prompt": prompt, "output": text.strip(),
                            "seconds": time.perf_counter() - started})
    result["generations"] = generations
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(v, spec=".4f", dash="—"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def delta(now, before, spec=".4f", lower_better=True):
    if now is None or before is None or (isinstance(now, float) and math.isnan(now)) \
            or (isinstance(before, float) and math.isnan(before)):
        return ""
    d = now - before
    if abs(d) < 1e-9:
        return "  (no change)"
    arrow = "improved" if (d < 0) == lower_better else "worse"
    return f"  ({d:+{spec.lstrip('.')}}, {arrow})"


def write_report(run_dir: Path, out_dir: Path, milestone: int, metrics: dict,
                 bench: dict, previous: dict | None, started_at: datetime) -> Path:
    cfg = metrics.get("config", {})
    step = metrics.get("step", milestone)
    total = cfg.get("total_steps", 1)
    tok_step = cfg.get("tokens_per_step", 0)
    now = datetime.now()

    prev_bench = (previous or {}).get("bench", {})
    prev_metrics = (previous or {}).get("metrics", {})

    L = []
    A = L.append
    A(f"# Milestone {milestone:,} — {now:%A %d %B %Y, %H:%M}")
    A("")
    A(f"Step **{step:,} of {total:,}** ({step/total*100:.1f}%). "
      f"{step*tok_step:,} of {cfg.get('corpus_tokens',0):,} tokens seen "
      f"({step*tok_step/max(1,cfg.get('corpus_tokens',1))*100:.1f}% of one pass).")
    A("")
    A(f"Wall clock since this log began: {str(now - started_at).split('.')[0]}.")
    A("")

    A("## Training state")
    A("")
    A("| Metric | Value | Change since last milestone |")
    A("|---|---|---|")
    A(f"| Training loss | {fmt(metrics.get('loss'))} |{delta(metrics.get('loss'), prev_metrics.get('loss'))} |")
    A(f"| Training perplexity | {fmt(math.exp(min(metrics.get('loss', 20), 20)), ',.1f')} | |")
    A(f"| Eval loss | {fmt(metrics.get('eval_loss'))} |{delta(metrics.get('eval_loss'), prev_metrics.get('eval_loss'))} |")
    A(f"| Learning rate | {fmt(metrics.get('lr'), '.2e')} | |")
    A(f"| Throughput | {fmt(metrics.get('tokens_per_second'), ',.0f')} tok/s | |")
    A(f"| MFU | {fmt((metrics.get('mfu') or 0)*100, '.1f')}% | |")
    A(f"| Peak VRAM | {fmt(metrics.get('peak_memory_gb'), '.2f')} GB | |")
    A(f"| GPU | {gpu_line()} | |")
    A("")

    A("## Held-out evaluation")
    A("")
    A("Measured on text the model has never trained on, on CPU, without interrupting training.")
    A("")
    A("| Metric | Value | Change |")
    A("|---|---|---|")
    A(f"| Nats per token | {fmt(bench.get('heldout_nats_per_token'))} |{delta(bench.get('heldout_nats_per_token'), prev_bench.get('heldout_nats_per_token'))} |")
    A(f"| Perplexity | {fmt(bench.get('heldout_perplexity'), ',.1f')} |{delta(bench.get('heldout_perplexity'), prev_bench.get('heldout_perplexity'), ',.1f')} |")
    A(f"| Bits per byte | {fmt(bench.get('heldout_bits_per_byte'))} |{delta(bench.get('heldout_bits_per_byte'), prev_bench.get('heldout_bits_per_byte'))} |")
    A("")
    code_n, prose_n = bench.get("code_nats"), bench.get("prose_nats")
    if code_n and prose_n and not math.isnan(code_n) and not math.isnan(prose_n):
        A(f"**Code vs prose:** code {fmt(code_n, '.3f')} nats, prose {fmt(prose_n, '.3f')} nats — "
          f"code is {'easier' if code_n < prose_n else 'harder'} by {abs(code_n-prose_n):.3f}. "
          "Code is lower-entropy text, so the model models it more cheaply; watching this gap "
          "narrow or widen shows whether prose modelling is catching up.")
        A("")

    A("## Generations")
    A("")
    A(f"Same ten prompts, same seed ({SEED}), temperature 0.7 at every milestone, so any "
      "difference is the model rather than sampling noise.")
    A("")
    prev_gens = {g["name"]: g["output"] for g in prev_bench.get("generations", [])}
    for gen in bench.get("generations", []):
        A(f"### `{gen['name']}`")
        A("")
        A(f"**Prompt:** `{gen['prompt']}`")
        A("")
        A("```")
        A(gen["output"] or "(empty)")
        A("```")
        if gen["name"] in prev_gens and prev_gens[gen["name"]] != gen["output"]:
            A("")
            A("<details><summary>previous milestone</summary>")
            A("")
            A("```")
            A(prev_gens[gen["name"]] or "(empty)")
            A("```")
            A("</details>")
        A("")

    A("## Notes")
    A("")
    A("- Benchmarks run on CPU from the latest checkpoint. Training was not paused, "
      "signalled or slowed on the GPU at any point.")
    A(f"- Model: {bench.get('params', 0):,} parameters, context {bench.get('block_size', '?')}.")
    A("- This is a base model: it continues text, it does not answer questions. "
      "Instruction following arrives with fine-tuning, after pretraining completes.")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"milestone-{milestone:05d}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def write_index(out_dir: Path, entries: list[dict]) -> None:
    L = ["# Geocentric 120M — training research log", "",
         "Automated checkpoints of a 120M-parameter model pretraining from scratch on a",
         "single RTX 2060. Each entry benchmarks the live checkpoint without interrupting",
         "the run.", "",
         "| Milestone | Time | Step | Loss | Held-out ppl | Report |",
         "|---|---|---|---|---|---|"]
    for e in entries:
        L.append(f"| {e['milestone']:,} | {e['time']} | {e['step']:,} | "
                 f"{fmt(e.get('loss'))} | {fmt(e.get('ppl'), ',.1f')} | "
                 f"[report](milestone-{e['milestone']:05d}.md) |")
    (out_dir / "INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/geocentric-120m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--heldout", default="runs/geocentric-120m/corpus/heldout_sample.txt")
    args = ap.parse_args()

    run_dir = (REPO / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out_dir = Path(args.out) if args.out else REPO / "research"
    started_at = datetime.now()

    # Held-out text: the tail of the corpus, which training never reaches in one pass.
    heldout_path = Path(args.heldout)
    if not heldout_path.exists():
        source = REPO / "data/pretrain/fineweb.txt"
        size = source.stat().st_size
        with source.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(max(0, size - 3_000_000))
            fh.readline()
            heldout = fh.read(600_000)
        heldout_path.parent.mkdir(parents=True, exist_ok=True)
        heldout_path.write_text(heldout, encoding="utf-8")
        log(f"held-out slice written to {heldout_path} ({len(heldout):,} chars)")
    heldout_text = heldout_path.read_text(encoding="utf-8")

    done: set[int] = {int(p.stem.split("-")[1]) for p in out_dir.glob("milestone-*.md")}
    entries: list[dict] = []
    # Deltas must survive a daemon restart, so the last benchmark is persisted.
    state_path = out_dir / "_state.json"
    previous: dict | None = None
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            previous = saved.get("previous")
            entries = saved.get("entries", [])
            log(f"restored delta baseline from {state_path.name}")
        except (OSError, json.JSONDecodeError):
            pass
    log(f"watching {run_dir}; milestones {MILESTONES}; already done {sorted(done) or 'none'}")

    while True:
        metrics = read_metrics(run_dir)
        step = metrics.get("step", 0)
        status = metrics.get("status", "")

        if not trainer_alive(run_dir) and status != "complete":
            log("WARNING: no training process found. Training appears to have stopped.")

        due = [m for m in MILESTONES if m not in done and step >= m]
        for milestone in due:
            log(f"milestone {milestone} reached (step {step}); benchmarking on CPU")
            try:
                bench = run_benchmark(run_dir, heldout_text, args.threads)
                path = write_report(run_dir, out_dir, milestone, metrics, bench, previous, started_at)
                previous = {"bench": bench, "metrics": dict(metrics)}
                entries.append({"milestone": milestone, "time": datetime.now().strftime("%d %b %H:%M"),
                                "step": step, "loss": metrics.get("loss"),
                                "ppl": bench.get("heldout_perplexity")})
                write_index(out_dir, entries)
                state_path.write_text(json.dumps({"previous": previous, "entries": entries},
                                                 indent=2, default=str), encoding="utf-8")
                done.add(milestone)
                log(f"wrote {path.name}")
            except Exception as exc:
                log(f"benchmark for milestone {milestone} failed: {exc!r}")
                done.add(milestone)  # do not spin on a milestone that cannot be produced

        if status == "complete" and all(m in done for m in MILESTONES if m <= step):
            log("training complete and all milestones written; exiting")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
