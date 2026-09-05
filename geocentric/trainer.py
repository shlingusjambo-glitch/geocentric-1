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
    quiet: bool = False,
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
    if quiet:
        return optimizer
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


def find_batch_size(
    model: nn.Module,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    max_batch: int = 64,
    safety: float = 0.85,
    probe_steps: int = 3,
) -> int:
    """Find the largest micro-batch that survives several real training steps.

    Analytic estimates do not survive contact with a real allocator: activation
    memory depends on the SwiGLU hidden width, what autograd chooses to save, and
    how fragmented the cache is. Measuring is cheap and is the difference between a
    run that starts and one that dies an hour in.

    The probe is non-destructive. It takes real optimizer steps on random data — so
    that optimizer state and fp16 unscaling are included in the measurement — then
    restores the original weights from a CPU snapshot and discards its throwaway
    optimizer. Probing with the caller's optimizer would leave the model trained on
    noise and its Adam moments seeded with garbage.

    Two further details: the budget comes from cuda.mem_get_info(), which excludes
    memory other processes (a desktop compositor, say) already hold; and more than
    one step is probed, because optimizer state is only allocated on the first step
    and steady-state usage is strictly higher.
    """
    if device.type != "cuda":
        return 4

    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = free_bytes * safety
    was_training = model.training
    model.train()
    vocab = int(getattr(model.config, "vocab_size", 32000))

    snapshot = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    probe_optimizer = build_optimizer(
        model, learning_rate, weight_decay, device_type="cuda", quiet=True
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    def fits(batch: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(probe_steps):
                ids = torch.randint(0, vocab, (batch, block_size), device=device)
                with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                    _, loss = model(ids, labels=ids)
                scaler.scale(loss).backward()
                scaler.step(probe_optimizer)
                scaler.update()
                probe_optimizer.zero_grad(set_to_none=True)
            return torch.cuda.max_memory_allocated() < budget
        except torch.OutOfMemoryError:
            return False
        finally:
            probe_optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    try:
        best = 0
        candidate = 1
        while candidate <= max_batch and fits(candidate):
            best = candidate
            candidate *= 2

        if best == 0:
            raise RuntimeError(
                f"A single sequence of {block_size} tokens does not fit in the "
                f"{free_bytes / 1024**3:.1f} GB free on this device. Reduce --block_size, "
                "choose a smaller --preset, or add --gradient_checkpointing."
            )

        low, high = best, min(best * 2, max_batch)
        while low + 1 < high:
            mid = (low + high) // 2
            if fits(mid):
                low = mid
            else:
                high = mid
    finally:
        # Undo everything the probe did before the real run starts.
        probe_optimizer.zero_grad(set_to_none=True)
        del probe_optimizer
        model.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
        del snapshot
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Auto batch size: {low} x {block_size} tokens (measured against {free_bytes / 1024**3:.1f} GB free)")
    return low


def estimate_batch_size(
    n_params: int,
    n_layer: int,
    n_embd: int,
    block_size: int,
    vram_gb: float,
    gradient_checkpointing: bool = False,
) -> int:
    """Coarse pre-flight guess used only before a model exists.

    Prefer find_batch_size(), which measures. This exists so the CLI can print a
    plan before allocating anything.
    """
    if vram_gb <= 0:
        return 4
    state_gb = n_params * 16 / 1024**3
    headroom_gb = vram_gb * 0.75 - state_gb
    if headroom_gb <= 0.25:
        return 1
    # The SwiGLU hidden layer is ~2.7x n_embd, and autograd saves both branches, so
    # per-token activation cost is far above the naive n_embd x n_layer figure.
    per_seq_bytes = block_size * n_embd * n_layer * 2 * 40
    if gradient_checkpointing:
        per_seq_bytes *= 0.25
    return max(1, min(64, int((headroom_gb * 1024**3) / max(1.0, per_seq_bytes))))
