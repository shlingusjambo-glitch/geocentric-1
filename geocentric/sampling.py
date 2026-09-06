"""Shared candidate-space sampling for batched and streamed generation."""
from __future__ import annotations

import torch


def sample_token(logits, history, temperature=0.8, top_k=50, top_p=0.95,
                 min_p=0.0, repetition_penalty=1.1, repetition_window=128,
                 processor=None):
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    logits = logits.float()
    if repetition_penalty != 1 and repetition_window > 0:
        recent = history[:, -repetition_window:]
        scores = logits.gather(1, recent)
        logits.scatter_(1, recent, torch.where(scores > 0, scores / repetition_penalty,
                                               scores * repetition_penalty))
    if temperature > 0:
        logits = logits / max(temperature, 1e-5)
    if processor is not None:
        logits = processor(logits, history)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    # Sorting 50 candidates instead of 32,000 vocabulary entries is material on
    # small models. Top-k is exact-k; ties at the cutoff follow torch.topk.
    candidates = None
    if 0 < top_k < logits.size(-1):
        logits, candidates = logits.topk(top_k, dim=-1)
    probabilities = logits.softmax(-1)
    if min_p > 0:
        probabilities = probabilities.masked_fill(
            probabilities < min_p * probabilities.amax(-1, keepdim=True), 0)
    if 0 < top_p < 1:
        ordered, indices = probabilities.sort(dim=-1, descending=True)
        keep = ordered.cumsum(-1) - ordered < top_p
        keep[:, 0] = True
        probabilities = torch.zeros_like(probabilities).scatter_(1, indices, ordered * keep)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(probabilities, 1)
    return sampled if candidates is None else candidates.gather(1, sampled)
