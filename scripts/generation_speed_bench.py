"""Matched greedy generation: previous full-capacity cache versus bounded buffers."""
import argparse
import json
import statistics
import time

import torch

from geocentric.model import GeocentricGPT, GPTConfig, KVCache
from geocentric.sampling import sample_token


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def previous(model, prompt, limit):
    caches = [KVCache(max_length=model.config.block_size) for _ in model.blocks]
    generated, cur, offset = prompt, prompt, 0
    for _ in range(limit):
        logits, _ = model(cur, caches=caches, position_offset=offset)
        offset += cur.size(1)
        cur = sample_token(logits[:, -1].float(), generated, temperature=0,
                           repetition_penalty=1)
        generated = torch.cat((generated, cur), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=16000, block_size=1024, n_layer=4,
                       n_head=4, n_kv_head=2, n_embd=256)
    model = GeocentricGPT(config).to(args.device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 64), device=args.device)
    dtype = torch.float16 if args.device == 'cuda' else torch.bfloat16
    timings = {'previous': [], 'buffered': []}
    outputs = {}
    for repeat in range(9):
        for name in list(timings)[::1 if repeat % 2 else -1]:
            sync(args.device); start = time.perf_counter()
            with torch.autocast(args.device, dtype=dtype, enabled=args.device != 'cpu'):
                outputs[name] = (previous(model, prompt, 64) if name == 'previous' else
                                 model.generate(prompt, 64, temperature=0, repetition_penalty=1))
            sync(args.device)
            if repeat:
                timings[name].append(time.perf_counter() - start)
    assert torch.equal(outputs['previous'], outputs['buffered'])
    rates = {name: 64 / statistics.median(times) for name, times in timings.items()}
    print(json.dumps({'device': args.device, 'torch': torch.__version__,
                      'tokens_per_second': rates, 'seconds': timings,
                      'improvement_percent': 100 * (rates['buffered'] / rates['previous'] - 1),
                      'caveat': 'Synthetic greedy batch generation, 64 prompt + 64 output tokens, '
                                '4 layers, width 256, 16000 vocabulary, context 1024. '
                                'Includes prefill. Not website streaming or RTX results unless run there.'}, indent=2))


if __name__ == '__main__':
    main()
