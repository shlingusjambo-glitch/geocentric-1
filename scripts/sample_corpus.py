"""Take an interleaved sample across corpus files, for tokenizer training.

Reading the first N bytes of one file would train the tokenizer on a single
source; sampling round-robin keeps the merge table representative of the mix the
model will actually see.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gb", type=float, default=2.0)
    ap.add_argument("--chunk_mb", type=float, default=16.0)
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in ([src] if src.is_file() else src.rglob("*")) if p.is_file() and p.suffix in {".txt", ".md"})
    if not files:
        raise SystemExit(f"No .txt/.md files under {src}")

    budget = int(args.gb * 1024**3)
    chunk = int(args.chunk_mb * 1024**2)
    per_file = max(1, budget // len(files))
    print(f"Sampling {args.gb:.1f} GB across {len(files)} file(s): {', '.join(f.name for f in files)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    handles = [f.open("r", encoding="utf-8", errors="replace") for f in files]
    taken = [0] * len(files)
    try:
        with out.open("w", encoding="utf-8") as sink:
            active = True
            while active and written < budget:
                active = False
                for i, handle in enumerate(handles):
                    if taken[i] >= per_file:
                        continue
                    piece = handle.read(chunk)
                    if not piece:
                        continue
                    sink.write(piece)
                    taken[i] += len(piece)
                    written += len(piece)
                    active = True
                    if written >= budget:
                        break
    finally:
        for handle in handles:
            handle.close()
    print(f"Wrote {written / 1024**3:.2f} GB to {out}")


if __name__ == "__main__":
    main()
