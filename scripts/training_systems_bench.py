#!/usr/bin/env python3
"""Reproducible systems ablation; random-token losses are NOT quality evidence.

Uses identical initialization and input tokens for every arm; timings synchronize
the accelerator. Warmup is excluded. CUDA reports actual allocator peak; MPS reports
only allocations sampled after forward/backward/update, not an unavailable peak.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from geocentric.device import select_device, resolve_dtype
from geocentric.epicycle import EpicycleConfig, EpicycleScheduler, RingAdamW
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.trainer import build_optimizer


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def allocated(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def run(args, name, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    torch.manual_seed(args.seed)
    model = GeocentricGPT(GPTConfig(vocab_size=args.vocab_size, block_size=args.context,
                                    n_layer=args.layers, n_head=8, n_kv_head=2,
                                    n_embd=args.width)).to(device)
    model.loss_chunk_size = 0 if name in {"dense", "fold_dense"} else args.chunk_size
    config = EpicycleConfig(enabled=True, deferent=False, horizon_start=args.context // 2,
                            equant=False)
    sched = EpicycleScheduler(config, model, 100, args.context)
    optimizer = (RingAdamW(model.parameters(), rings=4, dwell=2, factored=True)
                 if name == "capacity" else build_optimizer(model, 6e-4,
                                                            device_type=device.type, quiet=True))
    dtype = resolve_dtype(device, args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    generator = torch.Generator().manual_seed(args.seed + 1)
    data = [torch.randint(0, args.vocab_size, (args.batch, args.context + 1), generator=generator)
            for _ in range(args.steps + args.warmup)]
    seconds, losses, forward_bytes, sampled_bytes, states = [], [], [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, tokens in enumerate(data):
        if index == args.warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        x, y = tokens[:, :-1].to(device), tokens[:, 1:].to(device)
        if name in {"fold", "fold_dense", "capacity"}:
            x, y = sched.prepare_batch(x, y, args.context // 2)
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        started = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(x, labels=y, return_logits=False)
        sync(device)
        after_forward = allocated(device)
        scaler.scale(loss).backward()
        sync(device)
        after_backward = allocated(device)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradient in {name}; benchmark result would be invalid")
        scaler.step(optimizer)
        scaler.update()
        sync(device)
        elapsed = time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {name}")
        if index >= args.warmup:
            seconds.append(elapsed)
            losses.append(loss.item())
            forward_bytes.append(after_forward)
            sampled_bytes.append(max(after_forward, after_backward, allocated(device)))
            states.append(sum(t.numel() * t.element_size() for state in optimizer.state.values()
                              for t in state.values() if torch.is_tensor(t)))
    result = dict(arm=name, params=model.num_params(), median_step_seconds=statistics.median(seconds),
                  tokens=args.batch * args.context * args.steps,
                  tokens_per_second=args.batch * args.context * args.steps / sum(seconds),
                  max_forward_allocated_bytes=max(forward_bytes),
                  max_sampled_allocated_bytes=max(sampled_bytes),
                  cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if device.type == "cuda" else None),
                  max_optimizer_state_bytes=max(states), step_seconds=seconds,
                  losses=losses, dtype=str(dtype))
    print(json.dumps({k: v for k, v in result.items() if k not in {"losses", "step_seconds"}}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--arms", nargs="+", default=["dense", "chunked", "fold_dense", "fold", "capacity"],
                   choices=["dense", "chunked", "fold_dense", "fold", "capacity"])
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--output", default="runs/training_systems/results.json")
    args = p.parse_args()
    if min(args.steps, args.batch, args.context, args.width, args.layers, args.chunk_size) < 1:
        p.error("sizes and steps must be positive")
    device = select_device() if args.device == "auto" else torch.device(args.device)
    results = [run(args, arm, device) for arm in args.arms]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                   device=(torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)),
                   args=vars(args), results=results,
                   caveat="Random-token systems benchmark. No capability or convergence claims. "
                          "MPS sampled allocations are not peak memory. Arm order may affect thermals.")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
