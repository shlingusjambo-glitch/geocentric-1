#!/usr/bin/env python3
"""Compare current inference against a Git revision, with identical weights and inputs."""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.device import select_device, resolve_dtype


def sync(device):
    if device.type == 'mps': torch.mps.synchronize()
    if device.type == 'cuda': torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='3771334')
    p.add_argument('--device', default='auto')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', default='runs/inference-bench.json')
    args = p.parse_args()
    source = subprocess.check_output(['git', 'show', args.baseline + ':geocentric/model.py'])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'baseline.py'; path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('inference_baseline', path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline; spec.loader.exec_module(baseline)
    device = select_device() if args.device == 'auto' else torch.device(args.device)
    dtype = resolve_dtype(device, 'auto') if device.type != 'cpu' else torch.float32
    torch.manual_seed(2026)
    config = dict(vocab_size=32000, block_size=512, n_layer=6, n_head=6, n_kv_head=2, n_embd=384)
    current = GeocentricGPT(GPTConfig(**config)).to(device).eval()
    old = baseline.GeocentricGPT(baseline.GPTConfig(**config)).to(device).eval()
    old.load_state_dict(current.state_dict())
    ids = torch.randint(0, config['vocab_size'], (1, 64), device=device)
    results = {'baseline': [], 'current': []}
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        # Greedy output is invariant to cache allocation and candidate-space sampling.
        a = old.generate(ids, max_new_tokens=16, temperature=0)
        b = current.generate(ids, max_new_tokens=16, temperature=0)
        if not torch.equal(a, b): raise RuntimeError('Greedy equivalence failed')
        for trial in range(args.repeats + 1):
            for name, model in ([('baseline', old), ('current', current)] if trial % 2 == 0
                                else [('current', current), ('baseline', old)]):
                sync(device); start = time.perf_counter()
                model.generate(ids, max_new_tokens=args.tokens, temperature=.8, top_k=50,
                               top_p=.95, min_p=.05, repetition_penalty=1.25)
                sync(device)
                if trial: results[name].append(time.perf_counter() - start)
    rates = {name: args.tokens / statistics.median(times) for name, times in results.items()}
    payload = dict(args=vars(args), device=str(device), dtype=str(dtype), torch=torch.__version__,
                   config=config, seconds=results, median_tokens_per_second=rates,
                   improvement_percent=100*(rates['current']/rates['baseline']-1),
                   greedy_equal=True, caveat='Random weights; timing evidence only, not model quality.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2)+'\n'); print(json.dumps(payload, indent=2))


if __name__ == '__main__': main()
