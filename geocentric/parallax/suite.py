"""The PARALLAX probes and the runner that scores them."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from geocentric.chat import EOT, render_chat
from geocentric.parallax import probes
from geocentric.tokenizer_train import token_id

PARALLAX_VERSION = "1.0"

LN2 = math.log(2.0)


@dataclass
class ProbeResult:
    name: str
    score: Optional[float]              # 0-100, or None when the probe did not run
    headline: str
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def ran(self) -> bool:
        return self.score is not None


@dataclass
class SuiteResult:
    model_dir: str
    model_name: str
    stage: str
    params: int
    block_size: int
    vocab_size: int
    watermark: Optional[str]
    multimodal: bool
    device: str
    probes: List[ProbeResult]
    index: float
    weights: Dict[str, float]
    training: Dict[str, Any]
    seconds: float
    version: str = PARALLAX_VERSION

    def by_name(self, name: str) -> Optional[ProbeResult]:
        return next((p for p in self.probes if p.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


# ---------------------------------------------------------------------------
# Score curves
#
# Every probe reports a natural unit — bits per byte, accuracy, a ratio. Turning
# those into one number needs an explicit opinion about what counts as good, so the
# opinion lives here in one place instead of being smeared through the probes.
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_bpb(bpb: float) -> float:
    """Bits per byte -> 0-100, on a log scale between two real anchors.

    3.0 bpb is a model that has learned letter frequencies and not much else. 0.55
    is roughly where the best current open models sit on general English. A
    well-trained small model lands near 1.0, which this maps to about 60 — the curve
    is deliberately not generous, because the interesting question for a model this
    size is how far it still has to go.
    """
    if bpb <= 0 or not math.isfinite(bpb):
        return 0.0
    return 100.0 * _clamp(math.log(3.0 / bpb) / math.log(3.0 / 0.55))


def score_context_gain(gain_nats: float) -> float:
    """Nats of loss saved by having a full context instead of a fresh one.

    A healthy 1024-token model predicts the last quarter of a document a few tenths
    of a nat better than the first. Zero gain means the model is not using its
    context at all — which is exactly what the pre-3.0 line-shredding data bug
    produced, and the reason this probe exists.
    """
    return 100.0 * _clamp(gain_nats / 0.35)


def score_accuracy(accuracy: float, chance: float) -> float:
    """Accuracy above chance, rescaled so chance is 0 and perfect is 100."""
    return 100.0 * _clamp((accuracy - chance) / max(1e-6, 1.0 - chance))


def score_distinct(distinct3: float) -> float:
    """Distinct-trigram ratio -> 0-100. Below 0.2 is a loop; above 0.85 is healthy."""
    return 100.0 * _clamp((distinct3 - 0.20) / 0.65)


# ---------------------------------------------------------------------------
# Shared scoring machinery
# ---------------------------------------------------------------------------

@torch.no_grad()
def _token_losses(model, ids: Sequence[int], device: torch.device, block_size: int,
                  stride: Optional[int] = None,
                  skip_context: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token negative log likelihood over a long id sequence.

    Windows overlap by half. Only the second half of each window after the first is
    scored, so every scored token except the handful at the very start is predicted
    with at least block_size/2 tokens of context behind it. Scoring non-overlapping
    windows instead would give every 1024th token no context at all and quietly
    inflate the loss of short-context models less than long-context ones.

    `skip_context=False` scores every position in every window instead. MERIDIAN needs
    that: it buckets by position *within* the window, and with skipping on, the early
    buckets would only ever see the document's opening sentence — so the curve would be
    partly measuring "is the first paragraph unusual" rather than "does context help".
    Every other caller wants the skipping.

    Returns (losses, positions): the nll of each scored token and its offset within
    its window, which MERIDIAN needs and everything else ignores.
    """
    stride = stride or max(1, block_size // 2)
    losses: List[torch.Tensor] = []
    positions: List[torch.Tensor] = []
    ids_t = torch.tensor(ids, dtype=torch.long)
    n = ids_t.numel()
    if n < 2:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    start = 0
    while start < n - 1:
        window = ids_t[start : start + block_size + 1]
        if window.numel() < 2:
            break
        inp = window[:-1].unsqueeze(0).to(device)
        tgt = window[1:].unsqueeze(0).to(device)
        logits, _ = model(inp)
        nll = F.cross_entropy(
            logits[0].float(), tgt[0], reduction="none"
        )
        skip = 0 if (start == 0 or not skip_context) else min(nll.numel() - 1, block_size - stride)
        losses.append(nll[skip:].cpu())
        positions.append(torch.arange(skip, nll.numel()))
        if start + block_size + 1 >= n:
            break
        start += stride
    if not losses:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(losses), torch.cat(positions)


def _bits_per_byte(model, tokenizer, text: str, device: torch.device, block_size: int) -> Tuple[float, int, int]:
    ids = tokenizer.encode(text).ids
    losses, _ = _token_losses(model, ids, device, block_size)
    if losses.numel() == 0:
        return float("nan"), 0, 0
    n_bytes = len(text.encode("utf-8"))
    # Normalizing by bytes rather than tokens is what makes two models with
    # different vocabularies comparable: a bigger vocabulary spreads the same text
    # over fewer, individually harder tokens and looks worse per token while being
    # strictly better at modeling the text.
    return float(losses.sum()) / (LN2 * max(1, n_bytes)), losses.numel(), n_bytes


# ---------------------------------------------------------------------------
# ZENITH
# ---------------------------------------------------------------------------

def probe_zenith(model, tokenizer, device, block_size, slices: Dict[str, str]) -> ProbeResult:
    per_slice: Dict[str, float] = {}
    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    unknown = 0
    unk_id = tokenizer.token_to_id("<unk>")
    for name, text in slices.items():
        bpb, n_tokens, n_bytes = _bits_per_byte(model, tokenizer, text, device, block_size)
        per_slice[name] = round(bpb, 4)
        if math.isfinite(bpb):
            total_nats += bpb * LN2 * n_bytes
            total_bytes += n_bytes
            total_tokens += n_tokens
        # An <unk> is text the tokenizer cannot represent at all. Above a percent or
        # so, every score in this report is measuring the vocabulary rather than the
        # model, and saying so is more useful than a number.
        if unk_id is not None:
            unknown += sum(1 for i in tokenizer.encode(text).ids if i == unk_id)
    overall = total_nats / (LN2 * max(1, total_bytes))
    unk_rate = unknown / max(1, total_tokens)
    finite = {k: v for k, v in per_slice.items() if math.isfinite(v)}
    best = min(finite, key=finite.get) if finite else "-"
    worst = max(finite, key=finite.get) if finite else "-"
    spread = (finite[worst] - finite[best]) if finite else 0.0
    return ProbeResult(
        name="ZENITH",
        score=score_bpb(overall),
        headline=f"{overall:.3f} bits/byte",
        detail={
            "bits_per_byte": round(overall, 4),
            "by_slice": per_slice,
            "best_slice": best,
            "worst_slice": worst,
            "spread": round(spread, 4),
            "bytes_scored": total_bytes,
            "tokens_scored": total_tokens,
            "bytes_per_token": round(total_bytes / max(1, total_tokens), 2),
            "unk_rate": round(unk_rate, 4),
        },
        note=(
            (f"strongest on {best}, weakest on {worst} "
             f"({spread:.2f} bits/byte apart)" if finite else "no slice scored")
            + (f"; {unk_rate:.1%} of tokens were <unk> — the tokenizer cannot represent "
               "this text" if unk_rate > 0.01 else "")
        ),
    )


# ---------------------------------------------------------------------------
# MERIDIAN
# ---------------------------------------------------------------------------

def probe_meridian(model, tokenizer, device, block_size, document: str, buckets: int = 8) -> ProbeResult:
    ids = tokenizer.encode(document).ids
    if len(ids) < 64:
        return ProbeResult("MERIDIAN", None, "not run", note="document too short to bucket")

    stride = max(1, block_size // 4)
    losses, positions = _token_losses(model, ids, device, block_size, stride=stride,
                                      skip_context=False)
    if losses.numel() == 0:
        return ProbeResult("MERIDIAN", None, "not run", note="no scoreable tokens")

    width = max(1, block_size // buckets)
    curve: List[Optional[float]] = []
    for b in range(buckets):
        mask = (positions >= b * width) & (positions < (b + 1) * width)
        curve.append(round(float(losses[mask].mean()), 4) if bool(mask.any()) else None)

    present = [(i, v) for i, v in enumerate(curve) if v is not None]
    if len(present) < 2:
        return ProbeResult("MERIDIAN", None, "not run",
                           note=f"only {len(present)} position bucket(s) had data; "
                                "MERIDIAN needs a document longer than the context")
    first, last = present[0][1], present[-1][1]
    gain = first - last
    return ProbeResult(
        name="MERIDIAN",
        score=score_context_gain(gain),
        headline=f"{gain:+.3f} nats gained from context",
        detail={
            "bucket_width": width,
            "curve_nats": curve,
            "first_bucket": first,
            "last_bucket": last,
            "gain_nats": round(gain, 4),
            "document_tokens": len(ids),
        },
        note=(
            f"loss falls {gain:.3f} nats from the opening tokens to the end of the window"
            if gain > 0 else
            f"loss RISES {abs(gain):.3f} nats deeper into the context — the model is not using it"
        ),
    )


# ---------------------------------------------------------------------------
# SEXTANT
# ---------------------------------------------------------------------------

@torch.no_grad()
def _continuation_logprob(model, tokenizer, device, block_size, prefix: str, continuation: str) -> float:
    """Mean log-probability per token of `continuation` given `prefix`.

    Encoded together, not separately: byte-level BPE merges across the join, so
    encoding the two halves apart produces a token sequence the model never sees at
    inference and scores the wrong thing. Length-normalized because the candidates
    for a single item are not the same number of tokens and an unnormalized sum
    systematically prefers the shortest one.
    """
    prefix_ids = tokenizer.encode(prefix).ids
    full_ids = tokenizer.encode(prefix + continuation).ids
    n_prefix = len(prefix_ids)
    if len(full_ids) <= n_prefix:
        return -float("inf")
    # A merge across the boundary can shift where the prefix ends; walk back to the
    # last position where the two encodings still agree. Cloze prefixes are a handful
    # of tokens, so the quadratic worst case never bites.
    while n_prefix > 0 and full_ids[:n_prefix] != prefix_ids[:n_prefix]:
        n_prefix -= 1

    original_length = len(full_ids)
    full_ids = full_ids[-block_size:]
    n_prefix = max(1, n_prefix - (original_length - len(full_ids)))
    inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(inp)
    nll = F.cross_entropy(logits[0].float(), tgt[0], reduction="none")
    scored = nll[n_prefix - 1 :]
    if scored.numel() == 0:
        return -float("inf")
    return -float(scored.mean())


def probe_sextant(model, tokenizer, device, block_size, items) -> ProbeResult:
    correct = 0
    by_item: List[Dict[str, Any]] = []
    for prefix, answer, distractors in items:
        candidates = [answer, *distractors]
        scores = [_continuation_logprob(model, tokenizer, device, block_size, prefix, c)
                  for c in candidates]
        chosen = int(max(range(len(scores)), key=lambda i: scores[i]))
        hit = chosen == 0
        correct += int(hit)
        by_item.append({
            "prefix": prefix, "answer": answer,
            "chose": candidates[chosen], "correct": hit,
            "margin": round(scores[0] - max(scores[1:]), 4),
        })

    n = len(items)
    accuracy = correct / max(1, n)
    chance = 1.0 / (1 + len(items[0][2])) if n else 0.25
    misses = [i for i in by_item if not i["correct"]]
    return ProbeResult(
        name="SEXTANT",
        score=score_accuracy(accuracy, chance),
        headline=f"{correct}/{n} correct ({accuracy:.0%})",
        detail={
            "accuracy": round(accuracy, 4), "chance": round(chance, 4),
            "correct": correct, "total": n,
            "mean_margin": round(sum(i["margin"] for i in by_item) / max(1, n), 4),
            "misses": [f"{i['prefix']!r} -> {i['chose']!r} (wanted {i['answer']!r})"
                       for i in misses[:8]],
        },
        note=f"chance is {chance:.0%}; {len(misses)} item(s) missed",
    )


# ---------------------------------------------------------------------------
# ASTROLABE
# ---------------------------------------------------------------------------

@torch.no_grad()
def _answer(model, tokenizer, device, prompt: str, max_tokens: int) -> Tuple[str, bool]:
    """Greedy answer to one chat turn. Returns (text, stopped_on_its_own).

    Greedy, not sampled: a benchmark that changes its answer between runs is not a
    benchmark. The cost is that this measures the mode of the distribution rather
    than what a user at temperature 0.8 would see, which is the right trade for a
    format-compliance probe.
    """
    text, _ = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    ids = tokenizer.encode(text).ids[-model.config.block_size :]
    try:
        eot = token_id(tokenizer, EOT)
    except KeyError:
        eot = None
    out = model.generate(
        torch.tensor([ids], dtype=torch.long, device=device),
        max_new_tokens=max_tokens, temperature=0.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, eos_id=eot,
    )
    new = out[0, len(ids) :].tolist()
    stopped = eot is not None and eot in new
    if stopped:
        new = new[: new.index(eot)]
    return tokenizer.decode(new, skip_special_tokens=True).strip(), stopped


def _check(item: Dict[str, Any], answer: str, stopped: bool) -> bool:
    kind, arg = item["check"], item.get("arg")
    text = answer if item.get("case_sensitive") else answer.lower()
    if kind == "contains_word":
        needle = arg if item.get("case_sensitive") else str(arg).lower()
        return needle in text
    if kind == "word_count":
        return len(answer.split()) == int(arg)
    if kind == "min_lines":
        return len([l for l in answer.splitlines() if l.strip()]) >= int(arg)
    if kind == "starts_with_any":
        return any(text.lstrip().startswith(str(a).lower()) for a in arg)
    if kind == "shorter_than":
        return 0 < len(answer) < int(arg)
    if kind == "not_echo":
        prompt_words = set(item["prompt"].lower().split())
        answer_words = [w for w in text.split() if w]
        if not answer_words:
            return False
        overlap = sum(1 for w in answer_words if w in prompt_words) / len(answer_words)
        return overlap < 0.6
    if kind == "stops_cleanly":
        return stopped and bool(answer.strip())
    raise ValueError(f"Unknown ASTROLABE check {kind!r}")


def probe_astrolabe(model, tokenizer, device, items) -> ProbeResult:
    passed = 0
    rows: List[Dict[str, Any]] = []
    stops = 0
    empties = 0
    for item in items:
        answer, stopped = _answer(model, tokenizer, device, item["prompt"], item["max_tokens"])
        ok = _check(item, answer, stopped)
        passed += int(ok)
        stops += int(stopped)
        empties += int(not answer.strip())
        rows.append({"id": item["id"], "prompt": item["prompt"],
                     "answer": answer[:200], "pass": ok, "stopped": stopped})

    n = len(items)
    return ProbeResult(
        name="ASTROLABE",
        score=100.0 * passed / max(1, n),
        headline=f"{passed}/{n} instructions followed",
        detail={
            "passed": passed, "total": n,
            "stop_discipline": round(stops / max(1, n), 4),
            "empty_answers": empties,
            "results": rows,
        },
        note=(
            f"emitted a stop token on {stops}/{n} prompts"
            + (f"; {empties} answer(s) came back empty" if empties else "")
        ),
    )


# ---------------------------------------------------------------------------
# NADIR
# ---------------------------------------------------------------------------

def _distinct_n(tokens: Sequence[int], n: int) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def _longest_repeat_run(tokens: Sequence[int], n: int = 4) -> int:
    """Longest number of times any n-gram repeats back to back — a literal loop."""
    best = 0
    for period in range(1, min(24, len(tokens) // 2 + 1)):
        run = 0
        for i in range(len(tokens) - period):
            if tokens[i] == tokens[i + period]:
                run += 1
                best = max(best, run // period)
            else:
                run = 0
    return best


@torch.no_grad()
def probe_nadir(model, tokenizer, device, prompts, stage: str, max_tokens: int = 128) -> ProbeResult:
    torch.manual_seed(1729)  # reproducible sampling
    try:
        eot = token_id(tokenizer, EOT)
    except KeyError:
        eot = None

    rows: List[Dict[str, Any]] = []
    d1 = d2 = d3 = 0.0
    loops = 0
    for prompt in prompts:
        if stage == "sft":
            text, _ = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
        else:
            text = prompt
        ids = tokenizer.encode(text).ids[-model.config.block_size :]
        out = model.generate(
            torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=max_tokens, temperature=0.8, top_k=50, top_p=0.95,
            repetition_penalty=1.0,  # no penalty: the point is to see the raw failure
            eos_id=eot,
        )
        new = out[0, len(ids) :].tolist()
        if eot is not None and eot in new:
            new = new[: new.index(eot)]
        n1, n2, n3 = _distinct_n(new, 1), _distinct_n(new, 2), _distinct_n(new, 3)
        run = _longest_repeat_run(new)
        d1, d2, d3 = d1 + n1, d2 + n2, d3 + n3
        looped = run >= 3
        loops += int(looped)
        rows.append({"prompt": prompt, "tokens": len(new), "distinct_3": round(n3, 3),
                     "longest_loop": run, "sample": tokenizer.decode(new[:60], skip_special_tokens=True)})

    k = max(1, len(prompts))
    mean3 = d3 / k
    return ProbeResult(
        name="NADIR",
        score=score_distinct(mean3) * (1.0 - 0.5 * loops / k),
        headline=f"{mean3:.2f} distinct-3, {loops}/{k} looped",
        detail={
            "distinct_1": round(d1 / k, 4), "distinct_2": round(d2 / k, 4),
            "distinct_3": round(mean3, 4), "looped_generations": loops,
            "samples": rows,
        },
        note=(
            "no repetition loops at temperature 0.8" if loops == 0
            else f"{loops} of {k} generations fell into a repeating loop with no repetition penalty"
        ),
    )


# ---------------------------------------------------------------------------
# LODESTAR
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_lodestar(model, tokenizer, device, prompts, stage: str, max_tokens: int = 96) -> ProbeResult:
    """What the watermark buys, and what it costs, on this model.

    Every other generation probe here runs the model unmarked, because they exist to
    measure the model. This one exists to measure the mark, and it is the only place
    the tradeoff is quantified rather than asserted:

      detectability  the z-score the true identity recovers from a normal-length reply
      attribution    the z-score a decoy identity recovers from the same text (must be noise)
      cost           how far the bias pushed generation off the model's own preferred
                     path, in nats per token, measured by the model itself

    Each prompt is generated twice from the same seed, marked and unmarked, so the two
    differ only by the watermark bias and the comparison is not confounded by sampling.
    """
    from geocentric.watermark import (
        WatermarkConfig,
        WatermarkProcessor,
        detect_ids,
        watermark_capacity,
    )

    config = WatermarkConfig.from_dict(getattr(model.config, "watermark", None))
    if config is None or not config.enabled:
        return ProbeResult("LODESTAR", None, "not run",
                           note="this checkpoint is not watermarked")

    vocab = model.config.vocab_size
    processor = WatermarkProcessor(config, vocab)
    # Same gamma and delta, different name: this is the false-positive control.
    decoy = WatermarkConfig(identity=f"{config.identity}\x00decoy",
                            gamma=config.gamma, delta=config.delta)
    try:
        eot = token_id(tokenizer, EOT)
    except KeyError:
        eot = None

    def nll(prompt_ids: List[int], new_ids: List[int]) -> Optional[float]:
        if not new_ids:
            return None
        full = (prompt_ids + new_ids)[-model.config.block_size :]
        if len(full) < 2:
            return None
        logits, _ = model(torch.tensor([full[:-1]], dtype=torch.long, device=device))
        losses = F.cross_entropy(
            logits[0].float(), torch.tensor(full[1:], device=device), reduction="none"
        )
        return float(losses[-len(new_ids) :].mean())

    marked_z: List[float] = []
    last_marked: List[List[int]] = []
    plain_z: List[float] = []
    decoy_z: List[float] = []
    costs: List[float] = []
    lengths: List[int] = []

    for index, prompt in enumerate(prompts):
        if stage in {"sft", "vision"}:
            text, _ = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
        else:
            text = prompt
        ids = tokenizer.encode(text).ids[-model.config.block_size :]
        tensor = torch.tensor([ids], dtype=torch.long, device=device)

        runs = {}
        for label, proc in (("marked", processor), ("plain", None)):
            torch.manual_seed(4242 + index)  # identical sampling; only the bias differs
            out = model.generate(tensor, max_new_tokens=max_tokens, temperature=0.8,
                                 top_k=50, top_p=0.95, repetition_penalty=1.0,
                                 eos_id=eot, logits_processor=proc)
            new = out[0, len(ids) :].tolist()
            if eot is not None and eot in new:
                new = new[: new.index(eot)]
            runs[label] = new

        last_marked.append(runs["marked"])
        marked_z.append(detect_ids(runs["marked"], config, vocab, z_threshold=4.0).z_score)
        plain_z.append(detect_ids(runs["plain"], config, vocab, z_threshold=4.0).z_score)
        decoy_z.append(detect_ids(runs["marked"], decoy, vocab, z_threshold=4.0).z_score)
        lengths.append(len(runs["marked"]))
        a, b = nll(ids, runs["marked"]), nll(ids, runs["plain"])
        if a is not None and b is not None:
            costs.append(a - b)

    # Why the z came out where it did. A green-list mark can only hide in a choice the
    # model was not already certain about, so a confident model carries almost none —
    # and nothing in the output says so unless it is measured.
    capacity = None
    if lengths and max(lengths) >= 16:
        longest = max(range(len(prompts)), key=lambda i: lengths[i])
        sample = tokenizer.decode(last_marked[longest], skip_special_tokens=True)
        if sample.strip():
            capacity = watermark_capacity(model, tokenizer, sample, config)

    k = max(1, len(prompts))
    mean_marked = sum(marked_z) / k
    mean_plain = sum(plain_z) / k
    mean_decoy = sum(decoy_z) / k
    cost = sum(costs) / max(1, len(costs))
    mean_len = sum(lengths) / k

    from geocentric.watermark import MIN_SCORED_TOKENS

    if mean_len < MIN_SCORED_TOKENS:
        # The same rule the detector applies: below this length the z-test has no
        # power, so a number here would be a confident-looking measurement of noise.
        return ProbeResult(
            "LODESTAR", None,
            f"replies averaged {mean_len:.0f} tokens — too short to decide",
            detail={
                "identity": config.identity, "gamma": config.gamma, "delta": config.delta,
                "mean_generation_tokens": round(mean_len, 1),
                "minimum_tokens": MIN_SCORED_TOKENS, "samples": k,
            },
            note=(
                f"this model stops after ~{mean_len:.0f} tokens on these prompts, and the "
                f"z-test needs at least {MIN_SCORED_TOKENS}. The watermark is present and "
                "will be detectable in longer output; it cannot be measured here. Benchmark "
                "with prompts that elicit a paragraph, or detect on real generated text."
                + (f" Capacity on what it did produce: {capacity.verdict(config.gamma)}"
                   if capacity else "")
            ),
        )

    # z=6 is decisive; 0.5 nats/token of drift is a lot to pay for it.
    detect = _clamp(mean_marked / 6.0)
    penalty = _clamp(cost / 0.5)
    return ProbeResult(
        name="LODESTAR",
        score=100.0 * detect * (1.0 - 0.5 * penalty),
        headline=f"z={mean_marked:.1f} at {mean_len:.0f} tokens, {cost:+.3f} nats/token cost",
        detail={
            "identity": config.identity,
            "gamma": config.gamma, "delta": config.delta,
            "mean_z_true_identity": round(mean_marked, 3),
            "mean_z_decoy_identity": round(mean_decoy, 3),
            "mean_z_unmarked_text": round(mean_plain, 3),
            "quality_cost_nats_per_token": round(cost, 4),
            "mean_generation_tokens": round(mean_len, 1),
            "samples": k,
            "influenceable_fraction": (round(capacity.influenceable_fraction, 4)
                                       if capacity else None),
            "median_entropy_nats": round(capacity.median_entropy, 4) if capacity else None,
            "tokens_for_decisive_z": capacity.tokens_for_decisive_z if capacity else None,
            "capacity_verdict": capacity.verdict(config.gamma) if capacity else None,
        },
        note=(
            (f"detects at z={mean_marked:.1f} on ~{mean_len:.0f}-token replies"
             if mean_marked >= 4 else
             f"z={mean_marked:.1f} is below the decision threshold at this length — raise "
             f"delta above {config.delta} or expect only long passages to be attributable")
            + f"; a decoy identity reads z={mean_decoy:.1f} and unmarked text z={mean_plain:.1f}"
            + (f"; costs {cost:.3f} nats/token" if cost > 0.02 else "; no measurable quality cost")
            + (f". {capacity.verdict(config.gamma).capitalize()}" if capacity else "")
        ),
    )


# ---------------------------------------------------------------------------
# ORBIT
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_orbit(model, tokenizer, device, block_size) -> ProbeResult:
    prompt_ids = tokenizer.encode(probes.MERIDIAN_DOCUMENT).ids[: min(512, block_size - 64)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model(ids)  # warm the kernels; the first call on any backend is not representative
    if device.type == "cuda":
        torch.cuda.synchronize()
        # Otherwise this reports the high-water mark of every probe that ran before it.
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model(ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill = len(prompt_ids) / max(1e-6, time.perf_counter() - t0)

    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=64, temperature=0.0, repetition_penalty=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode = (out.size(1) - ids.size(1)) / max(1e-6, time.perf_counter() - t0)

    memory = 0.0
    if device.type == "cuda":
        memory = torch.cuda.max_memory_allocated(device) / 1024**3
    return ProbeResult(
        name="ORBIT",
        score=None,  # speed is not quality; reported, deliberately not scored
        headline=f"{decode:,.0f} tok/s decode",
        detail={
            "prefill_tokens_per_second": round(prefill, 1),
            "decode_tokens_per_second": round(decode, 1),
            "peak_memory_gb": round(memory, 3),
            "params": model.num_params(),
            "device": str(device),
        },
        note="throughput is reported but excluded from the Parallax Index — it measures "
             "the machine as much as the model",
    )


# ---------------------------------------------------------------------------
# PRISM
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_prism(model, tokenizer, device, vision_data: str, limit: int = 64) -> ProbeResult:
    """Is the model reading the image, or has it just learned caption-shaped text?

    Score each pair twice: once with its own image and once with another example's.
    A model that grounds its answer in the picture loses accuracy when the picture
    is wrong. A model that has memorized plausible caption prose scores identically
    both ways, and its captions look fine to a human reading them one at a time.
    That gap, not the raw loss, is the measurement.
    """
    from geocentric.vision import VisionConfig
    from geocentric.train_vision import VisionCollate, VisionSFTDataset

    vision = VisionConfig.from_dict(model.config.vision)
    dataset = VisionSFTDataset(tokenizer, vision_data, vision, model.config.block_size)
    n = min(limit, len(dataset))
    if n < 4:
        return ProbeResult("PRISM", None, "not run", note="need at least 4 image/text pairs")

    collate = VisionCollate(token_id(tokenizer, "<pad>"))
    matched: List[float] = []
    mismatched: List[float] = []
    for i in range(n):
        own = dataset[i]
        other = dataset[(i + n // 2) % n]
        if own["images"].shape != other["images"].shape:
            continue
        batch = collate([own])
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        for bank, images in (("m", own["images"]), ("x", other["images"])):
            _, loss = model(ids, labels=labels, images=images.to(device))
            (matched if bank == "m" else mismatched).append(float(loss))

    if not matched:
        return ProbeResult("PRISM", None, "not run", note="no comparable pairs (image counts differ)")
    mean_m = sum(matched) / len(matched)
    mean_x = sum(mismatched) / len(mismatched)
    gap = mean_x - mean_m
    # 0.15 nats is a modest but unambiguous signal that the pixels are being used.
    return ProbeResult(
        name="PRISM",
        score=100.0 * _clamp(gap / 0.30),
        headline=f"{gap:+.3f} nats grounding gap",
        detail={
            "matched_loss": round(mean_m, 4),
            "mismatched_loss": round(mean_x, 4),
            "gap_nats": round(gap, 4),
            "pairs": len(matched),
        },
        note=(
            f"captions cost {gap:.3f} nats more with the wrong image, so the model is "
            "using the pixels" if gap > 0.03 else
            "the wrong image costs the model nothing — it is generating caption-shaped "
            "text without looking at the picture"
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BASE_WEIGHTS = {"ZENITH": 0.40, "MERIDIAN": 0.25, "SEXTANT": 0.25, "NADIR": 0.10}
SFT_WEIGHTS = {"ZENITH": 0.25, "MERIDIAN": 0.15, "SEXTANT": 0.20, "ASTROLABE": 0.30, "NADIR": 0.10}
PRISM_WEIGHT = 0.20


def _weights(stage: str, ran: Sequence[str]) -> Dict[str, float]:
    base = dict(SFT_WEIGHTS if stage in {"sft", "vision"} else BASE_WEIGHTS)
    if "PRISM" in ran:
        for key in base:
            base[key] *= 1.0 - PRISM_WEIGHT
        base["PRISM"] = PRISM_WEIGHT
    # Drop probes that could not run and renormalize, so a skipped probe lowers
    # nothing — an index that silently counts a missing probe as zero is a lie.
    live = {k: v for k, v in base.items() if k in ran}
    total = sum(live.values()) or 1.0
    return {k: round(v / total, 4) for k, v in live.items()}


def run_parallax(
    model_dir: str | Path,
    eval_text: Optional[str] = None,
    vision_data: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_name: Optional[str] = None,
    nadir_tokens: int = 128,
) -> SuiteResult:
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.device import select_device
    from geocentric.training_metrics import load_training_metrics

    started = time.perf_counter()
    device = device or select_device()
    model, tokenizer, stage = load_model_and_tokenizer(
        model_dir, device=device, checkpoint_name=checkpoint_name, with_stage=True
    )
    model.eval()
    # A run benchmarked at partial depth would be measuring a model nobody ships.
    model.active_layers = None
    block_size = model.config.block_size

    wanted = {p.upper() for p in only} if only else None

    def enabled(name: str) -> bool:
        return wanted is None or name in wanted

    slices = dict(probes.ZENITH_SLICES)
    document = probes.MERIDIAN_DOCUMENT
    if eval_text:
        path = Path(eval_text).expanduser()
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else eval_text
        # Held-out text the user actually cares about beats the bundled sample every
        # time, so when it is supplied it replaces rather than supplements.
        slices = {"held-out": text}
        if len(text) > 4000:
            document = text

    results: List[ProbeResult] = []
    if enabled("ZENITH"):
        results.append(probe_zenith(model, tokenizer, device, block_size, slices))
    if enabled("MERIDIAN"):
        results.append(probe_meridian(model, tokenizer, device, block_size, document))
    if enabled("SEXTANT"):
        results.append(probe_sextant(model, tokenizer, device, block_size, probes.SEXTANT_ITEMS))
    if enabled("ASTROLABE"):
        if stage in {"sft", "vision"}:
            results.append(probe_astrolabe(model, tokenizer, device, probes.ASTROLABE_ITEMS))
        else:
            results.append(ProbeResult(
                "ASTROLABE", None, "not run",
                note="base checkpoint: it has never seen a chat template, so scoring it on "
                     "instruction following would measure the wrong thing",
            ))
    if enabled("NADIR"):
        results.append(probe_nadir(model, tokenizer, device, probes.NADIR_PROMPTS, stage, nadir_tokens))
    if enabled("LODESTAR") and getattr(model.config, "watermark", None):
        results.append(probe_lodestar(model, tokenizer, device, probes.NADIR_PROMPTS[:4], stage))
    if enabled("ORBIT"):
        results.append(probe_orbit(model, tokenizer, device, block_size))
    if enabled("PRISM"):
        if model.config.vision and vision_data:
            results.append(probe_prism(model, tokenizer, device, vision_data))
        elif model.config.vision:
            results.append(ProbeResult("PRISM", None, "not run",
                                       note="pass --vision_data with held-out image/text pairs"))

    ran = [r.name for r in results if r.ran]
    weights = _weights(stage, ran)
    index = sum(weights.get(r.name, 0.0) * (r.score or 0.0) for r in results)

    try:
        training = load_training_metrics(model_dir)
    except FileNotFoundError:
        training = {}

    return SuiteResult(
        model_dir=str(model_dir),
        model_name=model.config.model_name,
        stage=stage,
        params=model.num_params(),
        block_size=block_size,
        vocab_size=model.config.vocab_size,
        watermark=(model.config.watermark or {}).get("identity"),
        multimodal=bool(model.config.vision),
        device=str(device),
        probes=results,
        index=round(index, 2),
        weights=weights,
        training=training,
        seconds=round(time.perf_counter() - started, 2),
    )
