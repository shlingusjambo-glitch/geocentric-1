"""Fetch the pretraining and instruction data for the standard Geocentric recipe.

Data choice matters more than architecture at this scale. Three sources:

  fineweb-edu   Educational-quality filtered web text. Reaches a given quality on
                far fewer tokens than raw CommonCrawl, which is what makes a
                single-GPU budget viable at all.
  cosmopedia    Synthetic textbooks and stories. The Phi line of work showed that
                curated/synthetic prose punches well above its token count for
                small models; it is also what teaches clean, explanatory style.
  smol-smoltalk Instruction data filtered specifically for small models. Full-size
                chat sets are too hard for a 100M-parameter model and mostly teach
                it to imitate the shape of an answer it cannot produce.

Documents are written separated by a blank-line triple so `--doc_sep` can place a
real end-of-sequence marker at each boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOC_SEP = "\n\n\n"

PRETRAIN_SOURCES = {
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "field": "text",
        "share": 0.70,
    },
    "cosmopedia": {
        "repo": "HuggingFaceTB/cosmopedia-v2",
        "config": None,
        "split": "train",
        "field": "text",
        "share": 0.30,
    },
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def stream_to_text(name: str, spec: dict, target_bytes: int, out_path: Path, resume: bool) -> int:
    from datasets import load_dataset

    if resume and out_path.exists() and out_path.stat().st_size >= target_bytes:
        print(f"[{name}] already have {human(out_path.stat().st_size)} — skipping")
        return out_path.stat().st_size

    print(f"[{name}] streaming {spec['repo']} -> {out_path} (target {human(target_bytes)})")
    kwargs = {"split": spec["split"], "streaming": True}
    if spec["config"]:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(spec["repo"], **kwargs)

    written = 0
    documents = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for row in dataset:
            text = (row.get(spec["field"]) or "").strip()
            if len(text) < 200:
                continue
            sink.write(text)
            sink.write(DOC_SEP)
            written += len(text.encode("utf-8")) + len(DOC_SEP)
            documents += 1
            if documents % 20_000 == 0:
                pct = 100 * written / target_bytes
                print(f"[{name}] {documents:,} docs | {human(written)} | {pct:5.1f}%", flush=True)
            if written >= target_bytes:
                break
    print(f"[{name}] done: {documents:,} documents, {human(written)}")
    return written


def download_sft(out_path: Path, max_rows: int, resume: bool) -> int:
    from datasets import load_dataset

    if resume and out_path.exists() and out_path.stat().st_size > 0:
        rows = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"[sft] already have {rows:,} conversations — skipping")
        return rows

    print(f"[sft] streaming HuggingFaceTB/smol-smoltalk -> {out_path}")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for row in dataset:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            cleaned = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", "")).strip()}
                for m in messages
                if isinstance(m, dict) and str(m.get("content", "")).strip()
            ]
            if not any(m["role"] == "assistant" for m in cleaned):
                continue
            sink.write(json.dumps({"messages": cleaned}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 20_000 == 0:
                print(f"[sft] {kept:,} conversations", flush=True)
            if kept >= max_rows:
                break
    print(f"[sft] done: {kept:,} conversations")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--target_tokens", type=float, default=3.0e9,
                    help="Pretraining token budget. Text is fetched at ~4.2 bytes/token.")
    ap.add_argument("--bytes_per_token", type=float, default=4.2)
    ap.add_argument("--sft_rows", type=int, default=400_000)
    ap.add_argument("--only", default=None, help="Fetch just one of: fineweb, cosmopedia, sft")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    total_bytes = int(args.target_tokens * args.bytes_per_token)
    print(f"Target: {args.target_tokens / 1e9:.2f}B tokens ≈ {human(total_bytes)} of text\n")

    resume = not args.no_resume
    if args.only in (None, "fineweb", "cosmopedia"):
        for name, spec in PRETRAIN_SOURCES.items():
            if args.only and args.only != name:
                continue
            stream_to_text(name, spec, int(total_bytes * spec["share"]),
                           out / "pretrain" / f"{name}.txt", resume)

    if args.only in (None, "sft"):
        download_sft(out / "sft" / "smoltalk.jsonl", args.sft_rows, resume)

    print(f"\nReady. Pretraining text: {out / 'pretrain'}    SFT: {out / 'sft' / 'smoltalk.jsonl'}")


if __name__ == "__main__":
    main()
