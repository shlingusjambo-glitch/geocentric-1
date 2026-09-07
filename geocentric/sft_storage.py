"""Disk-backed SFT examples and bounded JSON parsing for desktop training."""
import json
import tempfile
from array import array

import numpy as np
import torch

MAX_RECORD = 16 * 1024 * 1024


def iter_json_records(path):
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as stream:
        if path.suffix.lower() == '.jsonl':
            while True:
                line = stream.readline(MAX_RECORD + 1)
                if not line:
                    return
                if len(line) > MAX_RECORD:
                    raise ValueError('SFT record exceeds 16 MiB; split oversized conversations')
                if line.strip():
                    yield json.loads(line)
            return
        buffer = ''
        eof = False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            eof = not chunk
            buffer += chunk
            if len(buffer) > MAX_RECORD:
                raise ValueError('SFT record exceeds 16 MiB; split oversized conversations')

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        is_array = buffer.startswith('[')
        if is_array:
            buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if is_array and first and buffer.startswith(']'):
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
                raise ValueError('SFT records must be JSON objects')
            yield value
            buffer = buffer[end:]
            first = False
            if not is_array:
                break
            whitespace()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                break
            if not buffer.startswith(','):
                raise ValueError('Expected comma between SFT JSON records')
            buffer = buffer[1:]
        whitespace()
        if buffer:
            raise ValueError('Trailing content after SFT JSON data')


class DiskExamples:
    """Store compact int32 token pairs; load only the requested example into RAM."""
    def __init__(self, directory=None):
        self._directory = tempfile.TemporaryDirectory(prefix='sft-data-', dir=directory)
        from pathlib import Path
        self.path = Path(self._directory.name) / 'examples.bin'
        self._writer = self.path.open('wb')
        self.offsets = array('Q', [0])

    def append(self, example):
        values = np.stack([example['input_ids'].numpy(), example['labels'].numpy()]).astype(np.int32)
        self._writer.write(values.tobytes())
        self.offsets.append(self.offsets[-1] + values.nbytes)

    def finish(self):
        self._writer.close()
        self._writer = None

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = self.offsets[index:index+2]
        with self.path.open('rb') as stream:
            stream.seek(begin)
            values = np.frombuffer(stream.read(end-begin), dtype=np.int32).reshape(2, -1)
        return {name: torch.from_numpy(values[i].astype(np.int64))
                for i, name in enumerate(('input_ids', 'labels'))}
