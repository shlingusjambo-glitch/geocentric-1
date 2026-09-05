"""A/B the legacy (2.1) training pipeline against the current one on identical data.

Both arms train from the same corpus slice for the same number of epochs, then are
scored on the same held-out text in bits per byte — a tokenizer-independent measure,
which matters because the arms use different vocabulary sizes and their per-token
losses are not comparable.

    python scripts/ab_compare.py --data data/wikitext103/wiki.train.tokens

The legacy arm is checked out from git history into a worktree; nothing in your
working tree is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The script lives in scripts/, so the package root is not on sys.path by default.
sys.path.insert(0, str(REPO))

LEGACY_COMMIT_DEFAULT = "77f2bb8"

# Legacy defaults, reproduced exactly as they shipped in 2.1.
LEGACY = {"vocab_size": 1024, "block_size": 64, "learning_rate": 3e-4}
CURRENT = {"vocab_size": 4096, "block_size": 256, "learning_rate": 6e-4}


def run(cmd: list[str], cwd: Path, log: Path) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n  (cwd={cwd}, log={log})")
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:  # type: ignore[union-attr]
            sink.write(line)
            if any(k in line for k in ("bits/byte", "architecture:", "Traceback", "Error", "error:")):
                print("   " + line.rstrip())
        return process.wait()


def prepare_data(source: Path, work: Path, corpus_mb: float, heldout_mb: float) -> tuple[Path, Path]:
    """Carve a training slice and a disjoint held-out slice from one corpus."""
    train_file = work / "train_slice.txt"
    heldout_file = work / "heldout_slice.txt"
    stamp_file = work / "slices.json"

    want_train = int(corpus_mb * 1024 * 1024)
    want_heldout = int(heldout_mb * 1024 * 1024)
    stamp = {"source": str(source), "train": want_train, "heldout": want_heldout}

    # Reuse only when the request matches; otherwise a changed --corpus-mb would
    # silently keep training on the previous slice.
    if train_file.exists() and heldout_file.exists() and stamp_file.exists():
        if json.loads(stamp_file.read_text(encoding="utf-8")) == stamp:
            print(f"Reusing existing slices in {work}")
            return train_file, heldout_file
        print("Slice request changed; re-cutting the corpus.")

    total = source.stat().st_size
    if total < want_train + want_heldout:
        raise SystemExit(
            f"{source} holds {total / 1024**2:.0f} MB but the run needs "
            f"{(want_train + want_heldout) / 1024**2:.0f} MB. Lower --corpus-mb."
        )

    print(f"Slicing {source.name}: {corpus_mb:.1f} MB train + {heldout_mb:.1f} MB held out")
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        train_text = handle.read(want_train)
        # Held-out text starts after the training slice, so nothing overlaps.
        heldout_text = handle.read(want_heldout)

    train_file.write_text(train_text, encoding="utf-8")
    heldout_file.write_text(heldout_text, encoding="utf-8")
    stamp_file.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return train_file, heldout_file


def ensure_legacy_tree(commit: str, path: Path) -> Path:
    """Check the legacy revision out into its own worktree, leaving HEAD untouched."""
    if (path / "geocentric" / "train_pretrain.py").exists():
        print(f"Reusing legacy worktree at {path}")
        return path

    # Deleting a worktree directory does not deregister it, so a rerun after cleanup
    # would otherwise fail with "already exists".
    subprocess.run(["git", "worktree", "prune"], cwd=str(REPO), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if path.exists():
        shutil.rmtree(path)

    print(f"Creating legacy worktree at {path} from {commit}")
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), commit],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Could not create the legacy worktree from {commit}:\n{result.stderr.strip()}\n\n"
            "Check that the commit exists (`git log --oneline`) and pass the right one "
            "with --legacy_commit."
        )
    return path


def fmt(value, spec: str = ".4f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def report(results: dict, out_dir: Path) -> str:
    legacy = results.get("legacy")
    current = results.get("current")
    lines: list[str] = []
    add = lines.append

    add("# Legacy (2.1) vs current training pipeline\n")
    add(f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    add("Both arms trained on the same corpus slice for the same number of epochs and")
    add("were scored on the same held-out text. **Bits per byte is the headline number**")
    add("— per-token loss is not comparable across arms because their vocabularies differ.\n")

    add("| Metric | Legacy 2.1 | Current | |")
    add("|---|---|---|---|")

    def row(label: str, key_path: list[str], spec: str = ".4f", lower_is_better: bool | None = None):
        def dig(arm):
            node = arm
            for key in key_path:
                if not isinstance(node, dict):
                    return None
                node = node.get(key)
            return node

        a, b = dig(legacy), dig(current)
        verdict = ""
        if lower_is_better is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            change = (b - a) / abs(a) * 100
            better = (b < a) if lower_is_better else (b > a)
            verdict = f"{'better' if better else 'worse'} by {abs(change):.1f}%"
        add(f"| {label} | {fmt(a, spec)} | {fmt(b, spec)} | {verdict} |")

    row("**Bits per byte** (held out)", ["heldout", "bits_per_byte"], ".4f", lower_is_better=True)
    row("Parameters", ["params"], ",", None)
    row("Vocabulary", ["vocab_size_actual"], ",", None)
    row("Context length", ["block_size_actual"], ",", None)
    row("Tokens per byte", ["heldout", "tokens_per_byte"], ".4f", None)
    row("Corpus tokens", ["corpus_tokens"], ",", None)
    row("Optimizer steps", ["steps"], ",", None)
    row("Training time (s)", ["train_seconds"], ",.0f", None)
    row("Throughput (tok/s)", ["tokens_per_second"], ",.0f", lower_is_better=False)
    row("Peak VRAM (GB)", ["peak_vram_gb"], ".2f", lower_is_better=True)
    row("Peak host RAM (GB)", ["peak_host_rss_gb"], ".2f", lower_is_better=True)

    if legacy and current:
        a = legacy.get("heldout", {}).get("bits_per_byte")
        b = current.get("heldout", {}).get("bits_per_byte")
        if a and b:
            add(f"\n## Verdict\n")
            delta = (a - b) / a * 100
            direction = "better" if b < a else "worse"
            add(f"The current pipeline is **{abs(delta):.1f}% {direction}** in bits per byte "
                f"({a:.4f} → {b:.4f}) on identical held-out text.\n")
            add(f"A byte of held-out text costs {a:.3f} bits under the legacy model and "
                f"{b:.3f} bits under the current one.\n")

    for name, arm in (("Legacy 2.1", legacy), ("Current", current)):
        if not arm:
            continue
        add(f"\n## {name} — sample completions\n")
        for item in arm.get("samples", []):
            add(f"**{item['prompt']}**\n")
            add(f"> {item['completion'].strip() or '(empty)'}\n")

    add("\n## Caveats\n")
    add("- Both arms are trained well below a compute-optimal token budget, so neither")
    add("  produces a good model. This measures which pipeline learns more from the")
    add("  same data, not what a finished model looks like.")
    add("- Each arm uses its own defaults end to end (vocabulary, context, learning")
    add("  rate, architecture). It answers \"what would I have gotten\", not which single")
    add("  change mattered most.")
    add("- Sample completions are from base models with no instruction tuning.")
    add("- Wall-clock time carries no verdict: the arms tokenize the same text into")
    add("  different numbers of tokens and therefore run different numbers of steps.")
    add("  Compare throughput (tokens/second) for speed.")
    if legacy and current:
        pa, pb = legacy.get("params"), current.get("params")
        if pa and pb and abs(pa - pb) / max(pa, pb) > 0.1:
            add(f"- **The arms are not the same size** ({pa:,} vs {pb:,} parameters), so this")
            add("  score partly reflects model capacity. Rerun with `--match-params` to")
            add("  build both at the same depth and width.")

    text = "\n".join(lines)
    (out_dir / "REPORT.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Raw text corpus, e.g. WikiText-103 train tokens")
    ap.add_argument("--work_dir", default="runs/ab_compare")
    ap.add_argument("--legacy_commit", default=LEGACY_COMMIT_DEFAULT)
    ap.add_argument("--preset", default="50m")
    ap.add_argument("--corpus-mb", type=float, default=200.0, dest="corpus_mb")
    ap.add_argument("--heldout-mb", type=float, default=2.0, dest="heldout_mb")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="Cap steps per arm (0 = full epochs)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--compile", dest="compile_mode", default="auto", choices=["auto", "off"])
    ap.add_argument("--arms", default="legacy,current")
    ap.add_argument("--match-params", action="store_true", dest="match_params",
                    help="Build both arms at the same depth and width, so the score "
                         "difference is not dominated by one arm being a larger model.")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    arms_requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.max_steps > 0 and "legacy" in arms_requested:
        raise SystemExit(
            "--max_steps caps only the current trainer; the 2.1 trainer has no such\n"
            "parameter and would run full epochs instead. The arms would then see\n"
            "different amounts of data and the comparison would be meaningless.\n\n"
            "Control the workload with --corpus-mb, which both arms honour."
        )

    work = (REPO / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    source = Path(args.data).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Corpus not found: {source}\nRun `geocentric download-wiki` first.")
    if source.is_dir():
        candidates = sorted(p for p in source.rglob("*") if p.suffix in {".txt", ".tokens"})
        if not candidates:
            raise SystemExit(f"No .txt/.tokens file under {source}")
        source = max(candidates, key=lambda p: p.stat().st_size)
        print(f"Using largest file in directory: {source}")

    train_file, heldout_file = prepare_data(source, work, args.corpus_mb, args.heldout_mb)

    shared_dims = None
    if args.match_params:
        from geocentric.param_compiler import compute_fluid_dimensions

        shared_dims = compute_fluid_dimensions(args.preset, vocab_size=CURRENT["vocab_size"])
        print(
            f"Size-matching both arms: {shared_dims['n_layer']} layers x "
            f"{shared_dims['n_embd']} wide x {shared_dims['n_head']} heads"
        )

    worker = REPO / "scripts" / "_ab_worker.py"
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results: dict = {}

    for arm in arms:
        settings = LEGACY if arm == "legacy" else CURRENT
        if arm == "legacy":
            tree = ensure_legacy_tree(args.legacy_commit, work / "legacy_tree")
            shutil.copy2(worker, tree / "scripts" / "_ab_worker.py")
        else:
            tree = REPO

        out_dir = work / f"{arm}_run"
        result_json = work / f"{arm}_result.json"
        cmd = [
            args.python, "scripts/_ab_worker.py",
            "--arm", arm,
            "--train_file", str(train_file),
            "--heldout_file", str(heldout_file),
            "--output_dir", str(out_dir),
            "--result_json", str(result_json),
            "--preset", args.preset,
            "--vocab_size", str(settings["vocab_size"]),
            "--block_size", str(settings["block_size"]),
            "--learning_rate", str(settings["learning_rate"]),
            "--epochs", str(args.epochs),
            "--max_steps", str(args.max_steps),
            "--batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--compile", args.compile_mode,
        ]
        if shared_dims:
            cmd += [
                "--n_layer", str(shared_dims["n_layer"]),
                "--n_head", str(shared_dims["n_head"]),
                "--n_embd", str(shared_dims["n_embd"]),
            ]
        code = run(cmd, tree, work / f"{arm}.log")
        if code != 0:
            log_text = (work / f"{arm}.log").read_text(encoding="utf-8", errors="replace")
            print(f"\n!! arm '{arm}' failed with exit code {code}. See {work / f'{arm}.log'}")
            if "ModuleNotFoundError" in log_text:
                missing = sorted({
                    line.split("'")[1] for line in log_text.splitlines()
                    if "ModuleNotFoundError" in line and "'" in line
                })
                # The legacy tree predates the dependency trim and needs packages the
                # current one dropped.
                print(f"   Missing dependency for this arm: {', '.join(missing)}")
                print(f"   Install it and rerun:  pip install {' '.join(missing)}")
            elif "ZeroDivisionError" in log_text and "lr_scheduler" in log_text:
                print("   The 2.1 trainer builds OneCycleLR with pct_start = int(steps*0.02)/steps,")
                print("   which collapses to a zero-length phase when a run lands between roughly")
                print("   50 and 99 optimizer steps. That is a pre-existing bug in that version,")
                print("   not a harness fault. Raise --corpus-mb so the run is comfortably longer.")
            else:
                print("   Last lines:")
                for line in log_text.strip().splitlines()[-8:]:
                    print("     " + line)
            print("   Continuing so the other arm still produces a result.")
            continue
        results[arm] = json.loads(result_json.read_text(encoding="utf-8"))

    if not results:
        raise SystemExit("Both arms failed; nothing to report.")

    (work / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(report(results, work))
    print("=" * 78)
    print(f"\nFull report: {work / 'REPORT.md'}")


if __name__ == "__main__":
    main()
