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
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
    watermark: Optional[WatermarkConfig] = None,
    images: Optional[torch.Tensor] = None,
) -> str:
    device = device or next(model.parameters()).device
    model.eval()

    ids = tokenizer.encode(prompt).ids
    # Keep the most recent context: the newest turn matters more than the oldest.
    ids = ids[-(model.config.block_size - max_new_tokens) :] or ids[-model.config.block_size :]
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
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
    watermark: Optional[WatermarkConfig] = None,
    images: Optional[torch.Tensor] = None,
) -> Iterator[str]:
    """Yield decoded text incrementally, reusing the KV cache between tokens."""
    from geocentric.model import KVCache

    device = device or next(model.parameters()).device
    model.eval()

    ids = tokenizer.encode(prompt).ids[-model.config.block_size :]
    stops = set(_stop_ids(tokenizer))
    caches = [KVCache() for _ in model.blocks]

    processor = make_processor(model, watermark)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    offset = 0
    produced: List[int] = []
    emitted = ""

    for _ in range(max_new_tokens):
        if offset + cur.size(1) > model.config.block_size:
            break
        logits, _ = model(cur, caches=caches, position_offset=offset,
                          images=images if offset == 0 else None)
        offset += cur.size(1)
        logits = logits[:, -1, :].float()

        history = torch.tensor([ids + produced], device=device)[:, -128:]
        if repetition_penalty != 1.0:
            score = torch.gather(logits, 1, history)
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            logits.scatter_(1, history, score)

        if temperature > 0:
            logits = logits / max(temperature, 1e-5)
        if processor is not None:
            # The green list is keyed on the tokens actually in the stream, prompt
            # included — the same view the detector reconstructs from the text.
            processor(logits, torch.tensor([ids + produced], device=device))

        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1))
        else:
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            probs = torch.softmax(logits, dim=-1)
            if min_p > 0:
                probs = torch.where(probs < min_p * probs.max(), torch.zeros_like(probs), probs)
            if 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
                keep[:, 0] = True
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs * keep)
            probs = probs / probs.sum().clamp_min(1e-9)
            next_id = int(torch.multinomial(probs, num_samples=1))

        if next_id in stops:
            break
        produced.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)

        # Decode the whole run each time so multi-byte characters never split.
        text = tokenizer.decode(produced, skip_special_tokens=True)
        if len(text) > len(emitted):
            yield text[len(emitted) :]
            emitted = text
