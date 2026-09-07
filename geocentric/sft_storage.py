"""Persistent disk-backed SFT examples and bounded JSON parsing."""
from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile

import numpy as np
import torch

MAX_RECORD = 16 * 1024 * 1024
CACHE_VERSION = 2


def iter_json_records(path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            while True:
                line = stream.readline(MAX_RECORD + 1)
                if not line:
                    return
                if len(line) > MAX_RECORD:
                    raise ValueError("SFT record exceeds 16 MiB; split oversized conversations")
                if line.strip():
                    yield json.loads(line)
            return
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            eof = not chunk
            buffer += chunk
            if len(buffer) > MAX_RECORD:
                raise ValueError("SFT record exceeds 16 MiB; split oversized conversations")

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        is_array = buffer.startswith("[")
        if is_array:
            buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if is_array and first and buffer.startswith("]"):
                buffer = buffer[1:]
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            if not isinstance(value, dict):
                raise ValueError("SFT records must be JSON objects")
            yield value
            buffer = buffer[end:]
            first = False
            if not is_array:
                break
            whitespace()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not buffer.startswith(","):
                raise ValueError("Expected comma between SFT JSON records")
            buffer = buffer[1:]
        whitespace()
        if buffer:
            raise ValueError("Trailing content after SFT JSON data")


def cache_fingerprint(data_path: Path, tokenizer, block_size: int, drop_overlong: bool) -> str:
    """Hash exact inputs so changed data/tokenizer settings never reuse stale tokens."""
    digest = hashlib.sha256()
    digest.update(f"sft-cache-{CACHE_VERSION}\0{block_size}\0{int(drop_overlong)}\0".encode())
    digest.update(tokenizer.to_str().encode())
    with data_path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


class DiskExamples:
    """Store compact int32 token pairs and persist a validated offset index."""

    def __init__(self, directory=None, key=None):
        self._mapping = None
        self._mapping_pid = None
        self.persistent = directory is not None and key is not None
        self._directory = None
        if self.persistent:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / f"{key}.bin"
            self.index_path = root / f"{key}.index.json"
            if self.path.is_file() and self.index_path.is_file():
                try:
                    metadata = json.loads(self.index_path.read_text())
                except (OSError, json.JSONDecodeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                offsets = metadata.get("offsets", [])
                dropped = metadata.get("dropped", 0)
                if (metadata.get("version") == CACHE_VERSION and isinstance(offsets, list)
                        and isinstance(dropped, int) and dropped >= 0 and len(offsets) > 1
                        and all(isinstance(x, int) and x >= 0 and x % 8 == 0 for x in offsets)
                        and offsets[0] == 0 and offsets[-1] == self.path.stat().st_size
                        and all(a < b for a, b in zip(offsets, offsets[1:]))):
                    self.offsets = array("Q", offsets)
                    self.dropped = dropped
                    self._writer = None
                    self.reused = True
                    return
            self._tmp_path = root / f"{key}.{secrets.token_hex(6)}.tmp"
        else:
            self._directory = tempfile.TemporaryDirectory(prefix="sft-data-", dir=directory)
            self.path = Path(self._directory.name) / "examples.bin"
            self._tmp_path = self.path
            self.index_path = None
        self.offsets = array("Q", [0])
        self._writer = self._tmp_path.open("wb")
        self.dropped = 0
        self.reused = False

    def append(self, example):
        if self.reused:
            raise RuntimeError("Cannot append to a completed SFT cache")
        values = np.stack([example["input_ids"].numpy(), example["labels"].numpy()]).astype(np.int32)
        self._writer.write(values.tobytes())
        self.offsets.append(self.offsets[-1] + values.nbytes)

    def finish(self, dropped=0):
        if self.reused:
            return
        self._writer.flush()
        self._writer.close()
        self._writer = None
        self.dropped = dropped
        if self.persistent:
            self._tmp_path.replace(self.path)
            metadata = {"version": CACHE_VERSION, "offsets": list(self.offsets), "dropped": dropped}
            temporary = self._tmp_path.with_suffix(".index.tmp")
            temporary.write_text(json.dumps(metadata))
            temporary.replace(self.index_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getstate__(self):
        # Spawn workers reopen the mapping rather than serializing its contents.
        state = self.__dict__.copy()
        state['_mapping'] = None
        state['_mapping_pid'] = None
        return state

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = self.offsets[index:index + 2]
        if self._mapping is None or self._mapping_pid != os.getpid():
            self._mapping = np.memmap(self.path, dtype=np.int32, mode='r')
            self._mapping_pid = os.getpid()
        values = self._mapping[begin // 4:end // 4].reshape(2, -1)
        return {name: torch.from_numpy(values[i].astype(np.int64))
                for i, name in enumerate(("input_ids", "labels"))}
