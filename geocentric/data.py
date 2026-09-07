from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from geocentric.chat import messages_from_record, render_chat
from geocentric.tokenizer_train import token_id

TEXT_SUFFIXES = {".txt", ".md"}
RECORD_SUFFIXES = {".jsonl", ".json", ".csv"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | RECORD_SUFFIXES

_STREAM_CHUNK = 1 << 20  # 1 MiB


def _resolve_data_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Training data path does not exist: {path}\n"
        "Point --data_path at a .txt/.md/.json/.jsonl/.csv file or a folder of them, "
        "or run `geocentric download-wiki` first."
    )


def _iter_files(p: Path) -> Iterator[Path]:
    if p.is_dir():
        found = False
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                found = True
                yield child
        if not found:
            raise ValueError(
                f"No supported training files under {p}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
    else:
        yield p


def _split_stream(handle, sep: str) -> Iterator[str]:
    """Stream a text file, splitting on `sep` without loading it all into memory."""
    buffer = ""
    while True:
        chunk = handle.read(_STREAM_CHUNK)
        if not chunk:
            break
        buffer += chunk
        pieces = buffer.split(sep)
        buffer = pieces.pop()
        for piece in pieces:
            if piece.strip():
                yield piece
    if buffer.strip():
        yield buffer


def iter_documents(path: str | Path, doc_sep: Optional[str] = None) -> Iterator[str]:
    """Yield whole documents — never individual lines.

    The previous implementation yielded raw text one line at a time and the dataset
    appended an end-of-sequence token after each, which trained the model to reset
    its context every ~15 tokens. That is why generations stayed locally fluent but
    never held a topic. Plain text files are now treated as a single continuous
    document unless `doc_sep` marks real boundaries; structured records are one
    document each, which they genuinely are.
    """
    root = _resolve_data_path(path)
    for file in _iter_files(root):
        suffix = file.suffix.lower()

        if suffix in TEXT_SUFFIXES:
            with file.open("r", encoding="utf-8", errors="replace") as handle:
                if doc_sep:
                    yield from _split_stream(handle, doc_sep)
                else:
                    while True:
                        chunk = handle.read(_STREAM_CHUNK * 8)
                        if not chunk:
                            break
                        yield chunk
            continue

        if suffix == ".jsonl":
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        text = record_to_text(json.loads(line))
                        if text:
                            yield text
            continue

        if suffix == ".json":
            payload = json.loads(file.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                text = record_to_text(row)
                if text:
                    yield text
            continue

        if suffix == ".csv":
            with file.open("r", encoding="utf-8", newline="") as handle:
                csv.field_size_limit(100_000_000)
                reader = csv.reader(handle)
                try:
                    first = next(reader)
                except StopIteration:
                    continue
                header_keys = {"instruction", "input", "output", "response", "text", "messages"}
                if any(isinstance(c, str) and c.lower() in header_keys for c in first):
                    for row in reader:
                        if len(row) <= len(first):
                            text = record_to_text({first[i]: row[i] for i in range(len(row))})
                            if text:
                                yield text
                else:
                    for row in [first, *reader]:
                        joined = "\n".join(row).strip()
                        if joined:
                            yield joined
            continue

        raise ValueError(f"Unsupported data format: {file}")


def iter_documents_with_source(
    path: str | Path, doc_sep: Optional[str] = None
) -> Iterator[Tuple[str, str]]:
    """Like iter_documents, but also names the file each document came from.

    prepare_corpus needs the boundaries between sources so it can hold out a slice
    of every one of them rather than a slice of whichever sorts last.
    """
    root = _resolve_data_path(path)
    for file in _iter_files(root):
        for doc in iter_documents(file, doc_sep=doc_sep):
            yield doc, str(file)


def record_to_text(row: Mapping[str, object]) -> str:
    """Flatten a structured record into pretraining text."""
    messages = messages_from_record(row)
    if messages:
        text, _ = render_chat(messages)
        return text
    text = str(row.get("text", "")).strip()
    if text:
        return text
    return "\n".join(f"{k}: {v}" for k, v in row.items()).strip()


# ---------------------------------------------------------------------------
# Binary token corpus
# ---------------------------------------------------------------------------

def _token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size < 2**16 else np.dtype(np.uint32)


def corpus_paths(out_dir: str | Path, split: str) -> Tuple[Path, Path]:
    out = Path(out_dir)
    return out / f"{split}.bin", out / f"{split}.meta.json"


def prepare_corpus(
    tokenizer: Tokenizer,
    data_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.005,
    doc_sep: Optional[str] = None,
    batch_size: int = 1024,
    progress: bool = True,
) -> Dict[str, int]:
    """Tokenize a corpus once into flat uint16/uint32 memmap files.

    Holding a whole corpus as a Python list of ints costs roughly 100 bytes per
    token, which capped training at a few tens of millions of tokens. A memmap of
    2-byte ids costs 2 bytes per token and never enters resident memory, so the
    corpus size is now bounded by disk rather than RAM.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab_size = tokenizer.get_vocab_size()
    dtype = _token_dtype(vocab_size)
    eos = token_id(tokenizer, "<eos>")

    train_bin, train_meta = corpus_paths(out, "train")
    val_bin, val_meta = corpus_paths(out, "val")

    # The validation split is a contiguous tail of the stream rather than random
    # windows. Random windows drawn from the same documents leak into training and
    # make eval loss look better than the model actually is.
    counts = {"train": 0, "val": 0, "documents": 0}
    buffer: List[str] = []

    total_written = 0
    tmp_bin = out / "corpus.tmp.bin"
    with tmp_bin.open("wb") as sink:
        def flush(batch: List[str]) -> None:
            nonlocal total_written
            if not batch:
                return
            for enc in tokenizer.encode_batch(batch):
                ids = enc.ids + [eos]
                sink.write(np.asarray(ids, dtype=dtype).tobytes())
                total_written += len(ids)

        # Record where each source file's tokens start and end, so the validation
        # split can take a tail from every source rather than a tail of the whole
        # stream. A single contiguous tail is whatever file happens to sort last:
        # adding html.txt silently made the validation set 100% markup, so eval
        # loss stopped measuring language at all.
        spans: List[Tuple[int, int, str]] = []
        span_start = 0
        current_source = None

        for doc, source in iter_documents_with_source(data_path, doc_sep=doc_sep):
            if source != current_source:
                if current_source is not None:
                    flush(buffer)
                    buffer = []
                    spans.append((span_start, total_written, current_source))
                    span_start = total_written
                current_source = source
            counts["documents"] += 1
            buffer.append(doc)
            if len(buffer) >= batch_size:
                flush(buffer)
                buffer = []
                if progress and counts["documents"] % (batch_size * 10) == 0:
                    print(f"  tokenized {counts['documents']:,} documents / {total_written:,} tokens")
        flush(buffer)
        if current_source is not None:
            spans.append((span_start, total_written, current_source))

    if total_written == 0:
        tmp_bin.unlink(missing_ok=True)
        raise ValueError(f"No tokens produced from {data_path}")

    # Take a token-disjoint tail of each source span. A document can straddle
    # the split; source metrics are validation loss, not proof of factual accuracy.
    val_ranges: List[Tuple[int, int]] = []
    train_ranges: List[Tuple[int, int]] = []
    source_meta = {"train": [], "val": []}
    offsets = {"train": 0, "val": 0}
    for start, end, source in spans or [(0, total_written, str(data_path))]:
        hold = int((end - start) * val_fraction)
        hold = max(0, min(hold, (end - start) // 2))
        for split, size in (("train", end - start - hold), ("val", hold)):
            if size:
                source_meta[split].append({"source": source, "start": offsets[split], "tokens": size})
                offsets[split] += size
        if hold:
            train_ranges.append((start, end - hold))
            val_ranges.append((end - hold, end))
        else:
            train_ranges.append((start, end))
    n_val = sum(e - s for s, e in val_ranges)
    n_train = sum(e - s for s, e in train_ranges)

    # Copy in chunks. np.asarray(memmap[:n]) would pull the whole split into RAM —
    # at a few billion tokens that is several GB and would fail on most machines.
    chunk_tokens = 64 << 20  # 128 MB at 2 bytes per token
    all_tokens = np.memmap(tmp_bin, dtype=dtype, mode="r", shape=(total_written,))
    for path, ranges in ((train_bin, train_ranges), (val_bin, val_ranges)):
        if not ranges:
            path.unlink(missing_ok=True)
            continue
        with path.open("wb") as sink:
            for begin, end in ranges:
                for offset in range(begin, end, chunk_tokens):
                    np.asarray(all_tokens[offset : min(offset + chunk_tokens, end)]).tofile(sink)
    del all_tokens
    tmp_bin.unlink(missing_ok=True)

    counts["train"] = n_train
    counts["val"] = n_val
    for path, count in ((train_meta, n_train), (val_meta, n_val)):
        if count > 0 or path is train_meta:
            path.write_text(
                json.dumps({"tokens": count, "dtype": dtype.name, "vocab_size": vocab_size,
                            "sources": source_meta["train" if path == train_meta else "val"]}, indent=2),
                encoding="utf-8",
            )
    if not n_val:
        val_meta.unlink(missing_ok=True)
    if progress:
        print(
            f"Corpus ready: {counts['documents']:,} documents -> "
            f"{n_train:,} train tokens, {n_val:,} val tokens ({dtype.name})"
        )
    return counts


class PackedDataset(Dataset):
    """Fixed-length windows over a memmapped token stream.

    Windows are cut from the concatenated corpus, so a training example can span a
    document boundary — which is what teaches long-range coherence. Nothing is
    copied until a batch is actually requested.
    """

    def __init__(self, bin_path: str | Path, block_size: int, dtype: str = "uint16") -> None:
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token corpus not found: {self.path}")
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        self.n_tokens = self.path.stat().st_size // self.dtype.itemsize
        self.n_windows = max(0, (self.n_tokens - 1) // block_size)
        if self.n_windows == 0:
            raise ValueError(
                f"Corpus {self.path} holds {self.n_tokens:,} tokens, too few for block_size={block_size}."
            )
        self._data: Optional[np.memmap] = None

    def _memmap(self) -> np.memmap:
        # Opened lazily so each DataLoader worker gets its own handle.
        if self._data is None:
            self._data = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._data

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.block_size
        window = np.asarray(self._memmap()[start : start + self.block_size + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(window[:-1]),
            "labels": torch.from_numpy(window[1:]),
        }


class SFTDataset(Dataset):
    """Multi-turn instruction data with everything but assistant turns masked out."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        path: str | Path,
        block_size: int,
        drop_overlong: bool = True,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        p = _resolve_data_path(path)
        from geocentric.sft_storage import DiskExamples, cache_fingerprint
        cache_key = cache_fingerprint(p, tokenizer, block_size, drop_overlong) if cache_dir else None
        self.examples = DiskExamples(cache_dir, cache_key)
        if self.examples.reused:
            self.dropped = self.examples.dropped
            print(f"SFT: reused {len(self.examples):,} cached tokenized conversations.", flush=True)
            return
        self.dropped = 0
        print(f"SFT: tokenizing {p.name} to persistent disk cache...", flush=True)
        for number, row in enumerate(self._read_rows(p), 1):
            messages = messages_from_record(row)
            if not messages or not any(m["role"] == "assistant" for m in messages):
                continue
            text, spans = render_chat(messages)
            # One whole conversation preserves BPE boundaries and assistant spans.
            enc = tokenizer.encode(text)
            if number % 1000 == 0:
                print(f"  SFT tokenization: {number:,} records, {len(self.examples):,} usable", flush=True)
            ids = enc.ids
            if len(ids) < 2:
                continue
            if len(ids) > block_size:
                # Truncating cuts the response mid-sentence and removes its end-of-turn
                # token, teaching the model never to stop. Dropping is the honest fix.
                if drop_overlong:
                    self.dropped += 1
                    continue
                ids = ids[:block_size]

            labels = [-100] * len(ids)
            for i, (tok_start, tok_end) in enumerate(enc.offsets[: len(ids)]):
                if tok_end <= tok_start:
                    continue
                for span_start, span_end in spans:
                    if tok_start >= span_start and tok_end <= span_end:
                        labels[i] = ids[i]
                        break

            if all(label == -100 for label in labels[1:]):
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                    "labels": torch.tensor(labels[1:], dtype=torch.long),
                }
            )

        self.examples.finish(self.dropped)
        if not self.examples:
            raise ValueError(f"No usable SFT examples found in {p}")
        if self.dropped:
            print(
                f"SFT: dropped {self.dropped:,} conversations longer than block_size={block_size} "
                "(truncating them would train the model never to emit an end-of-turn token)."
            )

    @staticmethod
    def _read_rows(path: Path) -> Iterator[Mapping[str, object]]:
        if path.suffix.lower() not in (".json", ".jsonl"):
            raise ValueError("SFT data must be .jsonl or .json")
        from geocentric.sft_storage import iter_json_records
        yield from iter_json_records(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def pad_collate(batch: Sequence[Mapping[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    if len(batch) == 1:
        # Default 6GB SFT microbatch: no padding or copies are needed.
        return {name: batch[0][name].unsqueeze(0) for name in ("input_ids", "labels")}
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for row, item in enumerate(batch):
        ids = item["input_ids"]
        lab = item["labels"]
        input_ids[row, :ids.numel()] = ids
        labels[row, :lab.numel()] = lab

    return {"input_ids": input_ids, "labels": labels}


class PadCollate:
    """Picklable collate function for DataLoader workers."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch):
        return pad_collate(batch, self.pad_id)
