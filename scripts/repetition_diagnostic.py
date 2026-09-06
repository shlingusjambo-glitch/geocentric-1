#!/usr/bin/env python3
"""Read-only repetition probe. Run against the affected checkpoint; does not modify training."""
import argparse
import json
from pathlib import Path
import torch
from geocentric.checkpoint import load_model_and_tokenizer
from geocentric.device import select_device, resolve_dtype
from geocentric.generate import stream_text, repeated_tail
from geocentric.model import KVCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--prompt', default='The purpose of learning is')
    p.add_argument('--device', default='auto')
    p.add_argument('--tokens', type=int, default=96)
    p.add_argument('--output', default='runs/repetition-diagnostic.json')
    args = p.parse_args()
    device = select_device() if args.device == 'auto' else torch.device(args.device)
    model, tokenizer, stage = load_model_and_tokenizer(args.model_dir, device=device,
        checkpoint_name=args.checkpoint, with_stage=True)
    dtype = resolve_dtype(device, 'auto') if device.type != 'cpu' else torch.float32
    ids = tokenizer.encode(args.prompt).ids[-model.config.block_size:]
    if len(ids) < 2: raise ValueError('Provide a prompt containing at least two tokens')
    runs = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        tensor = torch.tensor([ids], device=device)
        full, _ = model(tensor)
        caches = [KVCache(max_length=model.config.block_size) for _ in model.blocks]
        model(tensor[:, :-1], caches=caches)
        cached, _ = model(tensor[:, -1:], caches=caches, position_offset=len(ids)-1)
        error = (full[:, -1].float()-cached[:, -1].float()).abs().max().item()
        same_top1 = full[:, -1].argmax(-1).item() == cached[:, -1].argmax(-1).item()
        for temperature, penalty in [(0., 1.), (.8, 1.), (.8, 1.25)]:
            torch.manual_seed(2026)
            stats = {}
            text = ''.join(stream_text(model, tokenizer, args.prompt, max_new_tokens=args.tokens,
                temperature=temperature, repetition_penalty=penalty, stats=stats, loop_guard=False))
            generated = tokenizer.encode(text).ids
            runs.append(dict(temperature=temperature, repetition_penalty=penalty, text=text,
                has_sustained_loop=any(repeated_tail(generated[:i]) for i in range(6,len(generated)+1)),
                stats=stats))
    report = dict(checkpoint=args.checkpoint, model_dir=args.model_dir, stage=stage,
        prompt=args.prompt, device=str(device), dtype=str(dtype), cached_max_logit_error=error,
        cached_top1_equal=same_top1, runs=runs,
        interpretation='A sampled loop alone does not diagnose training collapse. Compare held-out loss, '
        'corpus duplication and base versus chat formatting. This probe never changes the checkpoint.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
