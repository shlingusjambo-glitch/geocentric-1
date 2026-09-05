#!/usr/bin/env python3
"""Measure what EPICYCLE actually buys, on your hardware, on your data.

A training technique that is only ever described is not a technique, it is a claim.
This script trains the same architecture on the same corpus once per arm and reports
the numbers side by side. It answers the only question that matters for a speed
mechanism — *how long until the loss reaches X* — rather than the question that is
easier to answer and means less, which is the loss after a fixed number of steps.

    python scripts/epicycle_bench.py --data data/corpus.txt --steps 400 --preset 50m

Every arm is evaluated at full depth and full context, so the comparison is between
finished models rather than between whatever each arm happened to be running at the
time. Output lands in runs/epicycle_bench/EPICYCLE_BENCH.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ARMS = {
    "baseline": "off",
    "speed": "speed",
    "quality": "quality",
    "memory": "memory",
    "full": "full",
    "capacity": "capacity",
}


def run_arm(name: str, preset_name: str, args) -> Dict[str, Any]:
    import random
    import numpy as np
    import torch
    from geocentric.train_pretrain import pretrain
    from geocentric.training_metrics import load_training_metrics

    out = Path(args.output) / name
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; choose a new --output or explicitly use --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    # Share one tokenizer and one tokenized corpus across arms. Retokenizing per arm
    # would put a few minutes of identical work inside every timing.
    shared = Path(args.output) / "_shared"
    if (shared / "tokenizer.json").exists():
        shutil.copyfile(shared / "tokenizer.json", out / "tokenizer.json")
        shutil.copytree(shared / "corpus", out / "corpus")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"\n{'=' * 70}\n  arm: {name}  (epicycle={preset_name})\n{'=' * 70}")
    started = time.perf_counter()
    pretrain(
        data_path=args.data,
        output_dir=str(out),
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        max_steps=args.steps,
        epochs=1000,  # let max_steps decide; the corpus may be small
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        save_every=0,
        dtype_name=args.dtype,
        compile_mode=args.compile_mode,
        modelver=f"epi_{name}",
        epicycle=preset_name,
        resume=False,
        num_workers=args.num_workers,
        loss_guard=False,  # recovery would introduce an uncontrolled fourth variable
        val_fraction=args.val_fraction,
    )
    wall = time.perf_counter() - started

    if not (shared / "tokenizer.json").exists():
        shared.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "tokenizer.json", shared / "tokenizer.json")
        shutil.copytree(out / "corpus", shared / "corpus")

    metrics = load_training_metrics(out)
    history = metrics.get("history", {})
    return {
        "arm": name,
        "preset": preset_name,
        "wall_seconds": round(wall, 1),
        "steps": metrics.get("step", 0),
        "final_loss": metrics.get("loss"),
        "eval_loss": metrics.get("eval_loss"),
        "tokens_seen": metrics.get("tokens_seen"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "peak_memory_gb": metrics.get("peak_memory_gb"),
        "events": metrics.get("epicycle_events", []),
        "history": {
            "steps": history.get("eval_steps", []),
            "loss": history.get("eval_loss", []),
            "elapsed": history.get("eval_elapsed", []),
        },
        "dir": str(out),
    }


def _smooth(values: List[float], window: int = 5) -> List[float]:
    """A raw loss curve is too noisy to read a crossing time off. Median-of-window."""
    out = []
    for i in range(len(values)):
        chunk = sorted(values[max(0, i - window + 1) : i + 1])
        out.append(chunk[len(chunk) // 2])
    return out


def time_to_loss(result: Dict[str, Any], target: float) -> Optional[float]:
    history = result["history"]
    losses, elapsed = history["loss"], history["elapsed"]
    if not losses or len(losses) != len(elapsed):
        return None
    for value, seconds in zip(_smooth(losses), elapsed):
        if value <= target:
            return round(seconds, 1)
    return None


def common_target(results: List[Dict[str, Any]]) -> Optional[float]:
    """The loss every arm actually reached, so the comparison is not extrapolated."""
    floors = []
    for r in results:
        losses = r["history"]["loss"]
        if not losses:
            return None
        floors.append(min(_smooth(losses)))
    ceiling = max(floors)
    return round(ceiling * 1.02, 4)  # 2% above the worst arm's floor: everyone gets there


def armillary_memory_probe(args) -> Dict[str, Any]:
    """Measure optimizer state directly rather than trusting the arithmetic."""
    import torch

    from geocentric.epicycle import RingAdamW
    from geocentric.model import GPTConfig, GeocentricGPT
    from geocentric.trainer import build_optimizer

    model = GeocentricGPT(GPTConfig(
        vocab_size=args.vocab_size, block_size=256, n_layer=args.n_layer,
        n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd,
    ))
    n_params = model.num_params()
    ids = torch.randint(0, args.vocab_size, (2, 64))

    def state_bytes(optimizer, steps: int) -> int:
        for _ in range(steps):
            model(ids, labels=ids)[1].backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
        total = 0
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.dim() > 0:
                    total += value.numel() * value.element_size()
        return total

    plain = state_bytes(build_optimizer(model, 1e-4, quiet=True, device_type="cpu"), 3)
    rows = [{"optimizer": "AdamW (fp32)", "rings": "-", "bytes": plain,
             "bytes_per_param": round(plain / n_params, 2)}]
    for rings in (1, 2, 4, 8):
        # dwell 1 so every ring is exercised inside a short probe; the ratio is the
        # same at the real dwell, only the momentum quality differs.
        measured = state_bytes(RingAdamW(model.parameters(), lr=1e-4, rings=rings, dwell=1), rings + 2)
        rows.append({
            "optimizer": "RingAdamW", "rings": rings, "bytes": measured,
            "bytes_per_param": round(measured / n_params, 2),
            # Weight and gradient are 4 bytes each and common to both, so they belong
            # in the denominator: quoting optimizer state alone overstates the win.
            "params_in_the_same_budget": round(
                (8 + plain / n_params) / (8 + measured / n_params), 2
            ),
        })
    return {"n_params": n_params, "rows": rows}


def render(results: List[Dict[str, Any]], memory: Dict[str, Any], args) -> str:
    baseline = next((r for r in results if r["arm"] == "baseline"), results[0])
    target = common_target(results)

    lines: List[str] = [
        "# EPICYCLE benchmark",
        "",
        f"`{args.data}` · {args.n_layer}L x {args.n_embd}d · ctx {args.block_size} · "
        f"{args.steps} optimizer steps per arm",
        "",
        "Each arm trains the same architecture on the same tokenized corpus for the same "
        "number of optimizer steps. Evaluation is at full depth and full context in every "
        "arm, so the losses compare finished models.",
        "",
        "## Results",
        "",
        "| arm | wall clock | speedup | tokens seen | final loss | eval loss | tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def cell(value: Optional[float], spec: str = ",.0f") -> str:
        return "—" if value is None else format(value, spec)

    for r in results:
        speedup = baseline["wall_seconds"] / max(1e-6, r["wall_seconds"])
        share = (r["tokens_seen"] or 0) / max(1, baseline["tokens_seen"] or 1)
        lines.append(
            f"| **{r['arm']}** | {r['wall_seconds']:,.0f}s | {speedup:.2f}x "
            f"| {cell(r['tokens_seen'])} ({share:.2f}x) "
            f"| {cell(r['final_loss'], '.4f')} | {cell(r['eval_loss'], '.4f')} "
            f"| {cell(r['tokens_per_second'])} |"
        )

    lines += [
        "",
        "HORIZON folds early windows into shorter independent sequences while retaining "
        "all targets. Compare total tokens as well as clock time; a final partial batch "
        "can contain fewer tokens than the nominal batch size.",
    ]

    # Flag the suspicious case in the output rather than in a footnote nobody reads.
    suspicious = [
        r for r in results
        if r is not baseline
        and r["wall_seconds"] < baseline["wall_seconds"]
        and r["final_loss"] is not None and baseline["final_loss"] is not None
        and r["final_loss"] < baseline["final_loss"]
        and (r["tokens_seen"] or 0) <= (baseline["tokens_seen"] or 0)
    ]
    if suspicious:
        names = ", ".join(f"`{r['arm']}`" for r in suspicious)
        lines += [
            "",
            f"> **Check this before believing it.** {names} finished faster *and* at a lower "
            "loss *and* on no more tokens. Winning on every axis at once is not what a "
            "speed/quality trade is supposed to look like, and the most common cause is a "
            "corpus that is easy enough to memorize — where reaching the floor sooner is all "
            "that is being measured. Confirm on real text before generalizing: the result is "
            "trustworthy exactly to the extent your benchmark corpus resembles what you will "
            "actually train on.",
        ]

    if target is not None:
        lines += [
            "",
            "## Time to reach held-out loss every arm reached",
            "",
            f"Target held-out loss **{target}** — 2% above the worst arm's floor, so no arm is being "
            "credited with a level it never touched. This is the number that matters: a "
            "technique that reaches the same loss sooner is faster, whatever it did to the "
            "step count.",
            "",
            "| arm | seconds to reach target | speedup |",
            "|---|---:|---:|",
        ]
        base_time = time_to_loss(baseline, target)
        for r in results:
            seconds = time_to_loss(r, target)
            if seconds is None:
                lines.append(f"| {r['arm']} | never reached | — |")
            elif base_time:
                lines.append(f"| {r['arm']} | {seconds:,.0f}s | **{base_time / seconds:.2f}x** |")
            else:
                lines.append(f"| {r['arm']} | {seconds:,.0f}s | — |")

    events = [r for r in results if r["events"]]
    if events:
        lines += ["", "## Gear changes", ""]
        for r in events:
            lines.append(f"- **{r['arm']}**: " + "; ".join(r["events"][:10]))

    lines += [
        "",
        "## ARMILLARY — optimizer state, measured",
        "",
        f"Model under test: {memory['n_params']:,} parameters. Byte counts are the actual "
        "allocated optimizer state after enough steps for every ring to have been hot.",
        "",
        "| optimizer | rings | state bytes | bytes/param | parameters in the same budget |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in memory["rows"]:
        ratio = row.get("params_in_the_same_budget")
        lines.append(
            f"| {row['optimizer']} | {row['rings']} | {row['bytes'] / 1024**2:,.1f} MB | "
            f"{row['bytes_per_param']:.2f} | {f'{ratio:.2f}x' if ratio else '1.00x (reference)'} |"
        )
    lines += [
        "",
        "The last column counts the 4 bytes per parameter for the weights and 4 for the "
        "gradient alongside the optimizer state, since all three are what a memory budget "
        "actually holds; quoting state in isolation would flatter the ratio. Momentum "
        "quality is the price: "
        "a parameter carries momentum for one ring period in every `rings`, and takes a "
        "momentum-free Adam step the rest of the time. Compare the `memory` arm above "
        "against `baseline` to see what that costs on this data before spending the "
        "headroom on a larger model.",
        "",
        "## Reading this honestly",
        "",
        "- One corpus, one architecture, one machine. A speedup here is evidence, not a law.",
        "- Depth growth and shorter conditioning change training. Repeat seeds and arm order; "
        "small improvements can be noise, and short runs do not measure final capability.",
        "- EQUANT changes what the loss is computed over during training. The reported loss "
        "is always the plain mean so the curves stay comparable, but the *gradient* is not "
        "the same, and its benefit shows up in downstream scores rather than in this table. "
        "Run `geocentric bench` on each arm to see it.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/epicycle_bench")
    parser.add_argument("--arms", nargs="+", default=["baseline", "speed", "memory"],
                        choices=list(ARMS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--preset", default=None,
                        help="Size the architecture from a parameter budget instead of "
                             "passing dimensions")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--compile", dest="compile_mode", default="off",
                        help="Left off by default: recompilation at each gear change would "
                             "be timed as if it were training")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--render_only", action="store_true",
                        help="Re-render the report from an existing results.json without "
                             "retraining. Use after editing the report format.")
    args = parser.parse_args()

    if args.preset:
        from geocentric.param_compiler import compute_fluid_dimensions

        dims = compute_fluid_dimensions(args.preset, vocab_size=args.vocab_size,
                                        block_size=args.block_size)
        args.n_layer, args.n_head = dims["n_layer"], dims["n_head"]
        args.n_kv_head, args.n_embd = dims["n_kv_head"], dims["n_embd"]
        print(f"Preset {args.preset}: {dims['n_layer']}L x {dims['n_embd']}d "
              f"= {dims['params']:,} params")

    out = Path(args.output)
    if args.render_only:
        saved = json.loads((out / "results.json").read_text())
        results, memory = saved["results"], saved["memory"]
        for key, value in saved["args"].items():
            if key not in {"render_only", "arms"}:
                setattr(args, key, value)
    else:
        # Prepare once, outside ALL timing arms. Otherwise the baseline pays the
        # tokenizer/corpus setup cost and appears slower for an unrelated reason.
        from geocentric.data import iter_documents, prepare_corpus
        from geocentric.tokenizer_train import train_byte_bpe_tokenizer
        shared = out / "_shared"
        if shared.exists():
            if not args.overwrite:
                raise FileExistsError(f"{shared} exists; use a fresh output or --overwrite")
            shutil.rmtree(shared)
        shared.mkdir(parents=True, exist_ok=True)
        tokenizer = train_byte_bpe_tokenizer(iter_documents(args.data), shared / "tokenizer.json",
                                              vocab_size=args.vocab_size)
        prepare_corpus(tokenizer, args.data, shared / "corpus", val_fraction=args.val_fraction)
        results = [run_arm(name, ARMS[name], args) for name in args.arms]
        print("\nMeasuring ARMILLARY optimizer state...")
        memory = armillary_memory_probe(args)

    report = out / "EPICYCLE_BENCH.md"
    report.write_text(render(results, memory, args), encoding="utf-8")
    if not args.render_only:
        (out / "results.json").write_text(
            json.dumps({"args": vars(args), "results": results, "memory": memory}, indent=2),
            encoding="utf-8",
        )
    print(f"\nWrote {report}")
    print(report.read_text())


if __name__ == "__main__":
    main()
