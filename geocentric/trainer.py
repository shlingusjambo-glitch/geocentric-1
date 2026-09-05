"""Shared training machinery: optimizer construction, LR schedule, throughput."""
from __future__ import annotations

import inspect
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from geocentric.device import device_flops


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "cuda",
) -> torch.optim.Optimizer:
    """AdamW with correct parameter groups and the fused kernel where available.

    Weight decay is applied only to matrices. Decaying norm gains and biases pulls
    them toward zero for no benefit and measurably hurts small models. The previous
    hand-rolled CPUAdamW decayed everything and ran an unfused Python loop.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    kwargs: Dict = {"lr": learning_rate, "betas": betas, "eps": 1e-8}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda":
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(groups, **kwargs)
    print(
        f"AdamW: {len(decay)} decayed tensors ({sum(p.numel() for p in decay):,} params), "
        f"{len(no_decay)} undecayed ({sum(p.numel() for p in no_decay):,})"
        f"{' [fused]' if kwargs.get('fused') else ''}"
    )
    return optimizer


def lr_at_step(
    step: int,
    total_steps: int,
    max_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay, driven by optimizer step count.

    Scheduling on steps rather than epochs means the schedule stays correct when a
    run is resumed, when the corpus grows, or when epochs is left open-ended.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return max_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_compile(model: nn.Module, device: torch.device, mode: str = "auto") -> Tuple[nn.Module, bool]:
    """Compile the model where it actually pays off."""
    if mode == "off" or not hasattr(torch, "compile"):
        return model, False
    if device.type != "cuda":
        # Inductor's CPU/MPS backends are slower than eager for this model size.
        return model, False
    try:
        # default mode rather than reduce-overhead: CUDA graphs interact badly with
        # gradient accumulation and with the KV cache during evaluation.
        compiled = torch.compile(model)
        print("torch.compile enabled.")
        return compiled, True
    except Exception as exc:
        print(f"torch.compile unavailable ({exc}); continuing eager.")
        return model, False


class Throughput:
    """Tokens/second and model FLOPs utilization."""

    def __init__(self, model_params: int, block_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.params = model_params
        self.block_size = block_size
        self.peak_flops = device_flops(device, dtype)
        self.reset()

    def reset(self) -> None:
        self.t0 = time.perf_counter()
        self.tokens = 0

    def add(self, tokens: int) -> None:
        self.tokens += tokens

    def read(self) -> Tuple[float, Optional[float]]:
        elapsed = max(1e-6, time.perf_counter() - self.t0)
        tps = self.tokens / elapsed
        mfu = None
        if self.peak_flops:
            # 6ND for forward+backward, plus the attention term.
            flops_per_token = 6 * self.params + 12 * self.block_size * self.params / max(1, self.params)
            mfu = (tps * flops_per_token) / self.peak_flops
        return tps, mfu


def format_progress(step: int, total: int, loss: float, lr: float, tps: float, mfu: Optional[float]) -> str:
    parts = [
        f"step {step}/{total}",
        f"loss {loss:.4f}",
        f"ppl {math.exp(min(loss, 20)):.1f}",
        f"lr {lr:.2e}",
        f"{tps:,.0f} tok/s",
    ]
    if mfu:
        parts.append(f"mfu {mfu * 100:.1f}%")
    return " | ".join(parts)


def estimate_batch_size(
    n_params: int, block_size: int, vram_gb: float, gradient_checkpointing: bool
) -> int:
    """Pick a micro-batch that fits, so the run does not die minutes in with OOM."""
    if vram_gb <= 0:
        return 4
    # Weights + Adam moments + grads in mixed precision, roughly 12 bytes per param.
    state_gb = n_params * 12 / 1024**3
    headroom = max(0.5, vram_gb * 0.85 - state_gb)
    per_seq_gb = (block_size * n_params ** 0.5 * 2e-6) / 1024
    if gradient_checkpointing:
        per_seq_gb *= 0.35
    batch = int(headroom / max(1e-6, per_seq_gb))
    return max(1, min(64, batch))
