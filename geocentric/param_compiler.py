"""Turn a parameter budget like '250m' into a balanced, hardware-sane architecture."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def parse_param_string(size_str: str) -> int:
    cleaned = str(size_str).strip().lower().replace("'", "").replace('"', "").replace("_", "")
    if cleaned.endswith("m"):
        return int(float(cleaned[:-1]) * 1_000_000)
    if cleaned.endswith("b"):
        return int(float(cleaned[:-1]) * 1_000_000_000)
    return int(float(cleaned))


def _ffn_hidden(n_embd: int) -> int:
    hidden = int(8 * n_embd / 3)
    return 128 * ((hidden + 127) // 128)


def exact_params(vocab_size: int, n_layer: int, n_embd: int, n_head: int, n_kv_head: int) -> int:
    """Parameter count for the real architecture, including weight tying."""
    head_dim = n_embd // n_head
    # Tied embeddings are one matrix, not two — the old compiler deducted it twice.
    embedding = vocab_size * n_embd
    attn = n_embd * n_embd * 2 + 2 * n_embd * (n_kv_head * head_dim)
    ffn = 3 * n_embd * _ffn_hidden(n_embd)
    norms = 2 * n_embd
    return embedding + n_layer * (attn + ffn + norms) + n_embd


def _kv_heads(n_head: int) -> int:
    # Grouped-query attention: roughly a quarter of the query heads, at least one,
    # and always an exact divisor. Shrinks the KV cache ~4x for negligible quality cost.
    for candidate in range(max(1, n_head // 4), 0, -1):
        if n_head % candidate == 0:
            return candidate
    return 1


def compute_fluid_dimensions(
    target_size_str: str,
    vocab_size: int = 32000,
    block_size: int | None = None,
) -> Dict[str, Any]:
    """Search width/depth pairs for the closest match to the parameter budget.

    Depth and width are chosen together against a target aspect ratio rather than
    read off a four-bucket ladder, and the result is verified against the exact
    parameter formula instead of a rough estimate.
    """
    target = parse_param_string(target_size_str)
    if target < 4_000_000:
        raise ValueError(f"Target size {target_size_str} is too small to be worth training.")

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for n_embd in range(256, 4097, 128):
        if n_embd % 64 != 0:
            continue
        n_head = max(4, n_embd // 64)
        if n_embd % n_head != 0:
            continue
        n_kv_head = _kv_heads(n_head)

        base = exact_params(vocab_size, 0, n_embd, n_head, n_kv_head)
        per_layer = exact_params(vocab_size, 1, n_embd, n_head, n_kv_head) - base
        n_layer = round((target - base) / per_layer)
        if n_layer < 4 or n_layer > 96:
            continue

        actual = exact_params(vocab_size, n_layer, n_embd, n_head, n_kv_head)
        size_error = abs(actual - target) / target
        if size_error > 0.15:
            continue

        # Well-shaped transformers sit near d_model/n_layer ~ 64-128. Penalize
        # both the very deep-and-thin and very wide-and-shallow ends.
        aspect = n_embd / n_layer
        aspect_error = abs(aspect - 96) / 96
        candidates.append((size_error + 0.5 * aspect_error, {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "n_embd": n_embd,
            "params": actual,
        }))

    if not candidates:
        raise ValueError(
            f"Could not find a balanced architecture near {target_size_str} for vocab {vocab_size}."
        )

    best = min(candidates, key=lambda item: item[0])[1]
    # 1024 minimum. The old default of 256 meant the model could never see enough
    # context to stay on a topic, no matter how long it trained.
    best["block_size"] = int(block_size) if block_size else (2048 if target >= 700_000_000 else 1024)
    best["intermediate_size"] = _ffn_hidden(best["n_embd"])
    return best


def recommended_tokens(n_params: int, ratio: int = 20) -> int:
    """Chinchilla-style compute-optimal token budget: about 20 tokens per parameter."""
    return int(n_params * ratio)
