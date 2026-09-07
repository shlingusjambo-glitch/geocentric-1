"""Matched SFT updates with masked prompts; measures systems speed, not quality."""
import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from geocentric.device import resolve_dtype, select_device
from geocentric.model import GeocentricGPT, GPTConfig


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--prompt_fraction', type=float, default=.5)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--output', default='runs/sft-speed.json')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or not 0 <= args.prompt_fraction < 1 or args.chunk_size < 1:
        parser.error('Invalid step count, prompt fraction, or chunk size')
    device = select_device() if args.device == 'auto' else torch.device(args.device)
    dtype = resolve_dtype(device) if device.type != 'cpu' else torch.float32
    torch.manual_seed(2026)
    config = GPTConfig(vocab_size=16000, block_size=512, n_layer=4, n_head=4,
                       n_kv_head=2, n_embd=256, gradient_checkpointing=True)
    baseline = GeocentricGPT(config).to(device)
    baseline.loss_chunk_size = args.chunk_size
    models = {'baseline': baseline, 'supervised_only': copy.deepcopy(baseline)}
    models['supervised_only'].loss_supervised_only = True
    optimizers = {k: torch.optim.AdamW(m.parameters(), lr=1e-4) for k, m in models.items()}
    scalers = {k: torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
               for k in models}
    timings = {k: [] for k in models}
    losses = {k: [] for k in models}
    for step in range(args.steps + args.warmup):
        x = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
        labels = x.roll(-1, 1)
        labels[:, :int(config.block_size * args.prompt_fraction)] = -100
        for name in list(models)[::1 if step % 2 == 0 else -1]:
            model, optimizer, scaler = models[name], optimizers[name], scalers[name]
            optimizer.zero_grad(set_to_none=True)
            sync(device); start = time.perf_counter()
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss = model(x, labels=labels, return_logits=False, loss_reduction='mean')[1]
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            sync(device); elapsed = time.perf_counter() - start
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite benchmark loss')
            if step >= args.warmup:
                timings[name].append(elapsed); losses[name].append(float(loss.detach()))
    rates = {k: config.block_size / statistics.median(v) for k, v in timings.items()}
    result = {'device': str(device), 'platform': platform.platform(), 'torch': torch.__version__,
              'dtype': str(dtype), 'config': vars(config), 'settings': vars(args),
              'tokens_per_second': rates, 'seconds': timings, 'losses': losses,
              'improvement_percent': 100 * (rates['supervised_only'] / rates['baseline'] - 1),
              'caveat': 'Synthetic full-model updates; both use activation checkpointing. '
                        'Total input tokens/s, not assistant tokens/s. Excludes data loading and saving. '
                        'Alternating order, identical initialization/batches; no quality claim.'}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('seconds', 'losses', 'config')}, indent=2))


if __name__ == '__main__':
    main()
