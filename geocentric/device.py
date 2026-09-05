from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RuntimeInfo:
    device: torch.device
    dtype: torch.dtype
    name: str
    total_memory_gb: float
    supports_bf16: bool
    supports_flash: bool


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    """True only where bfloat16 runs on tensor cores, not where it is emulated.

    torch.cuda.is_bf16_supported() answers True on Turing (sm_75) because bf16 is
    emulated there. Measured on an RTX 2060 that emulation runs at 3.2 TFLOPS
    against 24.3 for fp16 — picking bf16 on such a card costs roughly 7x
    throughput, so the check requires real hardware support.
    """
    if device.type == "cuda":
        try:
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            # Older torch has no such parameter; Ampere (sm_80) is the cutoff.
            return torch.cuda.get_device_properties(device).major >= 8
    if device.type == "mps":
        return True
    return hasattr(torch, "bfloat16")


def resolve_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    requested = (requested or "auto").lower()
    if requested in {"fp32", "float32"}:
        return torch.float32
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16

    # bfloat16 has the same exponent range as float32, so it needs no loss scaler and
    # will not produce the silent NaN cascades float16 does on a from-scratch model.
    if supports_bf16(device):
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def enable_fast_math() -> None:
    """Turn on the TF32 and matmul settings that cost accuracy we do not need."""
    # Expandable segments let the allocator grow a block instead of stranding
    # freed-but-unusable memory. On a 6 GB card that reclaims hundreds of MB.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def runtime_check(device: torch.device, dtype: torch.dtype) -> RuntimeInfo:
    name = "CPU"
    total = 0.0
    flash = False
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        name = props.name
        total = props.total_memory / 1024**3
        flash = props.major >= 8
    elif device.type == "mps":
        name = "Apple Silicon (MPS)"

    info = RuntimeInfo(
        device=device,
        dtype=dtype,
        name=name,
        total_memory_gb=total,
        supports_bf16=supports_bf16(device),
        supports_flash=flash,
    )
    mem = f", {total:.1f} GB VRAM" if total else ""
    print(f"Device: {name} ({device.type}{mem}) | dtype: {str(dtype).replace('torch.', '')}")
    if device.type == "cpu":
        print("WARNING: training on CPU. Expect this to be roughly 100x slower than a GPU.")
    return info


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**3
    return 0.0


def device_flops(device: torch.device, dtype: torch.dtype) -> Optional[float]:
    """Rough peak throughput used only for the model FLOPs utilization readout."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_properties(device).name.lower()
    table = {
        "h100": 989e12, "a100": 312e12, "l40": 181e12, "4090": 165e12, "4080": 97e12,
        "3090": 71e12, "3080": 59e12, "4070": 58e12, "3070": 40e12, "a10": 125e12,
        "2080": 40e12, "2070": 26e12, "2060": 26e12, "t4": 65e12,
    }
    for key, value in table.items():
        if key in name:
            return value if dtype != torch.float32 else value / 8
    return None


def cleanup(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# Retained for callers that still import the old name.
def cleanup_mps() -> None:
    cleanup(torch.device("mps"))
