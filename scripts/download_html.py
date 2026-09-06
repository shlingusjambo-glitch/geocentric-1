"""Fetch a high-quality HTML corpus for continued pretraining.

Source: hardikg2907/github-code-html-css — HTML and CSS files from GitHub.

webcode2m_purified has cleaner markup (121/121 well-formed against 86% here) but
embeds a full page screenshot in every parquet row. Those image bytes are read
whether or not the column is projected away, which measured at 40 KB/s of usable
text: roughly ten hours for this corpus against six minutes. The quality gap does
not justify a 100x cost, and filtering recovers most of it.

Filtering targets what a small model can actually learn from:

  * complete documents, not fragments — a page must open and close <html>
  * short enough to fit a context window or two, so the model sees whole
    structures rather than a middle slice of a 40 KB page
  * deduplicated by the dataset's own content hash
  * English, to avoid spending a limited token budget on markup wrapped around
    text the rest of the corpus never taught the model to read
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DOC_SEP = "\n\n\n"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretrain/html.txt")
    ap.add_argument("--target_tokens", type=float, default=500e6)
    ap.add_argument("--bytes_per_token", type=float, default=4.3)
    ap.add_argument("--min_chars", type=int, default=600)
    ap.add_argument("--max_chars", type=int, default=16000)
    ap.add_argument("--min_score", type=float, default=None)
    args = ap.parse_args()

    from datasets import load_dataset

    target = int(args.target_tokens * args.bytes_per_token)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Target {args.target_tokens/1e6:.0f}M tokens ≈ {human(target)} of HTML -> {out}")

    ds = load_dataset("hardikg2907/github-code-html-css", split="train", streaming=True)

    written = kept = seen = 0
    hashes: set = set()
    # A repeated <!DOCTYPE html> preamble appears in some records; collapse it so the
    # model does not learn to emit the declaration twice.
    dup_doctype = re.compile(r"(?i)(<!DOCTYPE html>\s*){2,}")
    # Hand-written HTML routinely separates sections with several blank lines, which
    # collides with the document separator. Left alone, a single page is split into
    # fragments and each fragment gets its own end-of-sequence token — teaching the
    # model that markup ends mid-element. Collapsing interior runs to two newlines
    # keeps the separator unique to real boundaries.
    blank_runs = re.compile(r"\n{3,}")

    with out.open("w", encoding="utf-8") as sink:
        for row in ds:
            seen += 1
            if "html" not in (row.get("language") or "").lower():
                continue
            text = (row.get("code") or "").strip()
            if not text:
                continue
            low = text.lower()
            if "<html" not in low or "</html>" not in low:
                continue
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            if args.min_score is not None and (row.get("score") or 0) < args.min_score:
                continue
            # Minified pages teach the model to emit one enormous line; skip them.
            if text.count("\n") < 5 or len(text) / max(1, text.count("\n") + 1) > 400:
                continue
            h = hash(text)
            if h is not None:
                if h in hashes:
                    continue
                hashes.add(h)

            text = dup_doctype.sub("<!DOCTYPE html>\n", text)
            text = blank_runs.sub("\n\n", text).strip()
            sink.write(text)
            sink.write(DOC_SEP)
            written += len(text.encode("utf-8")) + len(DOC_SEP)
            kept += 1
            if kept % 20_000 == 0:
                print(f"  {kept:,} pages kept of {seen:,} seen | {human(written)} | "
                      f"{written/target*100:5.1f}%", flush=True)
            if written >= target:
                break

    print(f"Done: {kept:,} pages kept from {seen:,} seen ({kept/max(1,seen)*100:.1f}% pass rate), "
          f"{human(written)} ≈ {written/args.bytes_per_token/1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
