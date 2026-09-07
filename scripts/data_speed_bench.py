"""Compare cached SFT loading/collation against the previous open/read path."""
import json
import statistics
import tempfile
import time

import numpy as np
import torch

from geocentric.data import pad_collate
from geocentric.sft_storage import DiskExamples


def legacy(cache, index):
    begin, end = cache.offsets[index:index + 2]
    with cache.path.open('rb') as stream:
        stream.seek(begin)
        values = np.frombuffer(stream.read(end - begin), dtype=np.int32).reshape(2, -1)
    example = {key: torch.from_numpy(values[i].astype(np.int64))
               for i, key in enumerate(('input_ids', 'labels'))}
    return {key: torch.stack([torch.cat((value, torch.empty(0, dtype=torch.long)))])
            for key, value in example.items()}


def main():
    with tempfile.TemporaryDirectory() as root:
        cache = DiskExamples(root, 'benchmark')
        for index in range(512):
            ids = torch.arange(512 + index % 513)
            cache.append({'input_ids': ids, 'labels': ids})
        cache.finish()
        timings = {'previous': [], 'mapped': []}
        for repeat in range(8):
            for name in list(timings)[::1 if repeat % 2 == 0 else -1]:
                start = time.perf_counter()
                for index in range(4096):
                    if name == 'previous':
                        result = legacy(cache, index % len(cache))
                    else:
                        result = pad_collate([cache[index % len(cache)]], 0)
                elapsed = time.perf_counter() - start
                if repeat:
                    timings[name].append(elapsed)
        rates = {name: 4096 / statistics.median(times) for name, times in timings.items()}
        print(json.dumps({'examples_per_second': rates, 'seconds': timings,
                          'improvement_percent': 100 * (rates['mapped'] / rates['previous'] - 1),
                          'caveat': 'Warm local file cache, batch one, 512 examples, 512–1023 tokens. '
                                    'CPU loading/collation only, not end-to-end training throughput.'}, indent=2))


if __name__ == '__main__':
    main()
