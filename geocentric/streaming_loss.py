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
