from __future__ import annotations

from typing import Iterator, List, Mapping, Optional, Sequence

import torch

from geocentric.chat import DEFAULT_SYSTEM, EOT, normalize_messages, render_chat
from geocentric.model import GeocentricGPT
from geocentric.tokenizer_train import token_id
from geocentric.watermark import WatermarkConfig, WatermarkProcessor


def make_processor(model: GeocentricGPT, watermark: Optional[WatermarkConfig] = None):
    """Build the logit processor for a model, defaulting to its own baked-in mark.

    A watermark that has to be requested at call time is a watermark that gets
    forgotten. The identity travels inside the checkpoint config, so any code path
    that generates from a marked model marks its output without being asked.
    """
    config = watermark or WatermarkConfig.from_dict(getattr(model.config, "watermark", None))
    if config is None or not config.enabled:
        return None
    return WatermarkProcessor(config, model.config.vocab_size)


def _stop_ids(tokenizer) -> List[int]:
    ids = []
    for name in (EOT, "<eos>"):
        try:
            ids.append(token_id(tokenizer, name))
        except KeyError:
            pass
    return ids


def build_chat_prompt(messages: Sequence[Mapping[str, str]], system: str = DEFAULT_SYSTEM) -> str:
    rows = normalize_messages(list(messages))
    if system and not any(m["role"] == "system" for m in rows):
        rows = [{"role": "system", "content": system}, *rows]
    text, _ = render_chat(rows, add_generation_prompt=True)
    return text


@torch.no_grad()
def generate_text(
    model: GeocentricGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repetition_penalty: float = 1.25,
    device: Optional[torch.device] = None,
    watermark: Optional[WatermarkConfig] = None,
    images: Optional[torch.Tensor] = None,
) -> str:
    device = device or next(model.parameters()).device
    model.eval()

    ids = tokenizer.encode(prompt).ids
    # Keep the most recent context: the newest turn matters more than the oldest.
    max_new_tokens = min(max(0, max_new_tokens), model.config.block_size - 1)
    ids = ids[-(model.config.block_size - max_new_tokens):] or [token_id(tokenizer, "<eos>")]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    stops = _stop_ids(tokenizer)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        eos_id=stops[0] if stops else None,
        repetition_penalty=repetition_penalty,
        logits_processor=make_processor(model, watermark),
        images=images,
    )
    new_ids = output[0, len(ids) :].tolist()
    for stop in stops:
        if stop in new_ids:
            new_ids = new_ids[: new_ids.index(stop)]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def stream_text(
    model: GeocentricGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repetition_penalty: float = 1.25,
    device: Optional[torch.device] = None,
    watermark: Optional[WatermarkConfig] = None,
    images: Optional[torch.Tensor] = None,
    cancel_event=None,
    stats: Optional[dict] = None,
    loop_guard: bool = False,
) -> Iterator[str]:
    """Stream with bounded KV/history allocations and cooperative cancellation."""
    import time
    from geocentric.model import KVCache
    from geocentric.sampling import sample_token

    device = device or next(model.parameters()).device
    model.eval()
    limit = min(max(0, max_new_tokens), model.config.block_size - 1)
    if limit <= 0:
        return
    encoded = tokenizer.encode(prompt).ids
    ids = encoded[-(model.config.block_size - limit):]
    if not ids:
        ids = [token_id(tokenizer, "<eos>")]
    if images is not None and len(ids) != len(encoded):
        raise ValueError("Image prompt is too long; shorten the conversation or output limit")
    stops = set(_stop_ids(tokenizer))
    caches = [KVCache(max_length=len(ids) + limit) for _ in model.blocks]
    processor = make_processor(model, watermark)
    history = torch.empty((1, len(ids) + limit), dtype=torch.long, device=device)
    history[:, :len(ids)] = torch.tensor([ids], dtype=torch.long, device=device)
    cur = history[:, :len(ids)]
    offset = 0
    produced: List[int] = []
    emitted = ""
    started = time.perf_counter()
    reason = "length"
    first_token_time = None
    for _ in range(limit):
        if cancel_event is not None and cancel_event.is_set():
            reason = "cancelled"
            break
        logits, _ = model(cur, caches=caches, position_offset=offset,
                          images=images if offset == 0 else None)
        offset += cur.size(1)
        next_id = sample_token(logits[:, -1, :], history[:, :len(ids) + len(produced)],
                               temperature, top_k, top_p, min_p, repetition_penalty,
                               processor=processor)
        token = int(next_id.item())
        if first_token_time is None:
            first_token_time = time.perf_counter() - started
        if token in stops:
            reason = "stop"
            break
        history[:, len(ids) + len(produced):len(ids) + len(produced) + 1] = next_id
        produced.append(token)
        cur = next_id
        text = tokenizer.decode(produced, skip_special_tokens=True)
        # A byte-BPE token may end halfway through a Unicode character. Do not
        # permanently emit the decoder's temporary replacement character.
        if not text.endswith("\ufffd") and text.startswith(emitted) and len(text) > len(emitted):
            yield text[len(emitted):]
            emitted = text
        if loop_guard and repeated_tail(produced):
            reason = "repetition"
            break
    text = tokenizer.decode(produced, skip_special_tokens=True)
    if text.startswith(emitted) and len(text) > len(emitted):
        yield text[len(emitted):]
    if stats is not None:
        elapsed = time.perf_counter() - started
        stats.update(prompt_tokens=len(ids), generated_tokens=len(produced),
                     truncated_prompt_tokens=max(0, len(encoded) - len(ids)),
                     seconds=elapsed, tokens_per_second=len(produced) / max(elapsed, 1e-9),
                     first_token_seconds=first_token_time, finish_reason=reason)


def repeated_tail(tokens, repeats=6, max_period=4):
    """Detect sustained exact token cycles, not ordinary repeated words elsewhere.

    Six consecutive copies of a 1–4 token phrase trigger a visible stop reason.
    This is a serving guard, not evidence that training divergence was diagnosed.
    """
    for width in range(1, max_period + 1):
        if len(tokens) >= width * repeats:
            pattern = tokens[-width:]
            if tokens[-width * repeats:] == pattern * repeats:
                return True
    return False
