"""Exact token-chunked linear cross entropy, portable to CPU, MPS and CUDA.

Checkpoint the projection *and* CE together: merely chunking the forward retains
every chunk's softmax for backward. This bounds vocabulary-sized intermediates by
chunk_size * vocab_size, while retaining a global per-token loss for EQUANT.
This is rematerialization, not a new loss or a fused Cut Cross-Entropy kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def supervised_cross_entropy(hidden, weight, labels, chunk_size=256, reduction="sum", selected=None):
    """Project only supervised positions; attention still sees the whole prompt.

    Intended for SFT sum/mean loss. Ignored prompt/padding rows have zero gradient
    under ordinary CE, so removing their vocabulary projections preserves the
    objective. Keep a differentiable zero for entirely masked microbatches.
    Optional selected indices must be the complete ordered nonignored positions
    in flattened labels, on the same device. The SFT loader constructs these.
    """
    if reduction not in {"sum", "mean"} or torch.compiler.is_compiling():
        return linear_cross_entropy(hidden, weight, labels, chunk_size, reduction)
    flat = hidden.reshape(-1, hidden.size(-1))
    targets = labels.reshape(-1)
    if flat.size(0) != targets.numel() or targets.numel() == 0 or chunk_size < 1:
        raise ValueError("Invalid token shapes or chunk size")
    # SFT can select rows on CPU before transfer, avoiding CUDA nonzero's
    # data-dependent output allocation and associated host synchronization.
    if selected is None:
        selected = (targets != -100).nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        return flat.sum() * 0 + weight.reshape(-1)[:1].sum() * 0
    return linear_cross_entropy(flat.index_select(0, selected), weight,
                                targets.index_select(0, selected), chunk_size, reduction)


def linear_cross_entropy(hidden, weight, labels, chunk_size=256, reduction="mean"):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unsupported loss reduction: {reduction}")
    hidden = hidden.reshape(-1, hidden.size(-1))
    labels = labels.reshape(-1)
    if hidden.size(0) != labels.numel() or labels.numel() == 0:
        raise ValueError("hidden and labels must have the same nonzero token count")

    def project_loss(h, w, target):
        return F.cross_entropy(F.linear(h, w).float(), target,
                               ignore_index=-100, reduction="none")

    losses = []
    for start in range(0, labels.numel(), chunk_size):
        args = (hidden[start:start + chunk_size], weight, labels[start:start + chunk_size])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            loss = checkpoint(project_loss, *args, use_reentrant=False,
                              preserve_rng_state=False)
        else:
            loss = project_loss(*args)
        losses.append(loss)
    per_token = torch.cat(losses)
    if reduction == "none":
        return per_token
    total = per_token.sum()
    if reduction == "sum":
        return total
    # An entirely masked microbatch produces a differentiable zero, not NaN.
    return total / (labels != -100).sum().clamp_min(1)


class _SparseReplayCE(torch.autograd.Function):
    """Replay only rows with a nonzero upstream gradient (e.g. EQUANT's band).

    Forward still scores every token. Backward is first-order only and retains
    PyTorch's linear/CE kernels, including the forward autocast precision.
    """
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.device_type = hidden.device.type
        ctx.amp_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.amp_dtype = torch.get_autocast_dtype(ctx.device_type)
        return torch.cat([F.cross_entropy(
            F.linear(hidden[start:start + chunk_size], weight).float(),
            labels[start:start + chunk_size], ignore_index=-100, reduction="none")
            for start in range(0, labels.numel(), chunk_size)])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hidden, weight, labels = ctx.saved_tensors
        dh = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        # Compact once over the entire batch, then replay full-size chunks. This
        # avoids running many underfilled matrix multiplications for scattered rows.
        selected = ((upstream != 0) & (labels != -100)).nonzero(as_tuple=True)[0]
        for start in range(0, selected.numel(), ctx.chunk_size):
            indices = selected[start:start + ctx.chunk_size]
            with torch.enable_grad(), torch.autocast(ctx.device_type, dtype=ctx.amp_dtype,
                                                      enabled=ctx.amp_enabled):
                h = hidden.index_select(0, indices).detach().requires_grad_(True)
                w = weight.detach().requires_grad_(True)
                losses = F.cross_entropy(F.linear(h, w).float(), labels[indices],
                                         ignore_index=-100, reduction="none")
                grad_h, grad_w = torch.autograd.grad(losses, (h, w), upstream[indices])
            if dh is not None:
                dh.index_copy_(0, indices, grad_h)
            if dw is not None:
                dw.add_(grad_w)
        return dh, dw, None, None


def sparse_replay_cross_entropy(hidden, weight, labels, chunk_size=256):
    """Per-token CE with selected-row backward replay; same first-order objective.

    Dynamic row compaction is an eager optimization. Compiled graphs use the
    established checkpointed implementation rather than depending on dynamic
    output-shape support in a particular compiler/backend release.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    hidden = hidden.reshape(-1, hidden.size(-1))
    labels = labels.reshape(-1)
    if labels.numel() == 0 or hidden.size(0) != labels.numel():
        raise ValueError("hidden and labels must have the same nonzero token count")
    if torch.compiler.is_compiling() or not torch.is_grad_enabled():
        return linear_cross_entropy(hidden, weight, labels, chunk_size, "none")
    return _SparseReplayCE.apply(hidden, weight, labels, chunk_size)
