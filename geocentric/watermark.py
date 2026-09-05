"""Statistical watermarking: mark generated text so it can be attributed later.

The scheme is the green-list construction of Kirchenbauer et al. (2023). At every
decoding position the previous `context_width` tokens are hashed together with a key
derived from the model's *identity string*; that hash seeds a PRNG which splits the
vocabulary into a "green" fraction and a "red" remainder, and green logits get a
small bias. Nothing about the text looks unusual, but green tokens appear far more
often than the 25% chance would predict, and a one-sided z-test recovers that.

Keying the split on the identity is what makes this an attribution mechanism rather
than a bare "AI or not" flag: detecting with the right identity gives a large z, and
detecting the same text under any other identity gives noise. So `detect()` also
answers *whose* model wrote it, given a list of candidates.

Two honest limits, stated because a watermark that is oversold is worse than none:

- Detection needs enough tokens. Below ~40 scored tokens the z-test cannot separate
  a watermark from chance no matter how strong the bias.
- Heavy paraphrasing removes it. This survives light editing, not rewriting.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

WATERMARK_FILE = "watermark.json"
_SALT = "geocentric-watermark-v1"


@dataclass
class WatermarkConfig:
    """How a model marks its own output.

    gamma: fraction of the vocabulary that is green at each position.
    delta: bias added to green logits, in post-temperature logit units so the
           strength does not silently change when the user changes temperature.
    context_width: how many preceding tokens seed the split. 1 is the most robust
           to editing; larger values are harder to reverse-engineer but break as
           soon as any earlier token in the window changes.
    """

    identity: str
    gamma: float = 0.25
    delta: float = 2.0
    context_width: int = 1
    enabled: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.identity).strip():
            raise ValueError("Watermark identity cannot be empty — it is the attribution key.")
        self.identity = str(self.identity).strip()
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")
        if self.context_width < 1:
            raise ValueError(f"context_width must be at least 1, got {self.context_width}")

    @property
    def key(self) -> int:
        digest = hashlib.sha256(f"{_SALT}\x00{self.identity}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["WatermarkConfig"]:
        if not data or not data.get("identity"):
            return None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / WATERMARK_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> Optional["WatermarkConfig"]:
        path = Path(directory) / WATERMARK_FILE
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


@lru_cache(maxsize=8192)
def _green_mask_cpu(key: int, prefix: Tuple[int, ...], vocab_size: int, gamma: float) -> torch.Tensor:
    """Boolean mask over the vocabulary, deterministic in (key, prefix).

    A Bernoulli draw rather than a permutation: `randperm(32000)` per generated token
    is the dominant cost of a naive implementation, while one uniform vector is a
    single kernel. The green set is then binomial around gamma*V instead of exactly
    gamma*V, which costs nothing — the detector applies the identical rule, so the
    null hypothesis it tests against is still exactly gamma.
    """
    seed = hashlib.sha256(
        key.to_bytes(8, "big") + b"".join(int(t).to_bytes(8, "big", signed=True) for t in prefix)
    ).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(seed[:8], "big"))
    return torch.rand(vocab_size, generator=generator) < gamma


class WatermarkProcessor:
    """Applies the green-list bias to a logit row during decoding."""

    def __init__(self, config: WatermarkConfig, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size
        self._device_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._device: Optional[torch.device] = None

    def _mask(self, prefix: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self._device != device:
            self._device_cache.clear()
            self._device = device
        cached = self._device_cache.get(prefix)
        if cached is None:
            cached = _green_mask_cpu(self.config.key, prefix, self.vocab_size, self.config.gamma).to(device)
            if len(self._device_cache) > 4096:
                self._device_cache.clear()
            self._device_cache[prefix] = cached
        return cached

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """logits: (B, V) for the next position. generated: (B, T) tokens so far."""
        if not self.config.enabled or self.config.delta == 0.0:
            return logits
        width = self.config.context_width
        if generated.size(1) < width:
            return logits
        prefixes = generated[:, -width:].tolist()
        for row, prefix in enumerate(prefixes):
            logits[row] = logits[row] + self.config.delta * self._mask(tuple(prefix), logits.device)
        return logits


@dataclass
class DetectionResult:
    identity: str
    z_score: float
    p_value: float
    green_tokens: int
    scored_tokens: int
    green_fraction: float
    watermarked: bool
    reason: str

    def summary(self) -> str:
        verdict = "WATERMARKED" if self.watermarked else "no watermark detected"
        return (
            f"{verdict} | identity={self.identity!r} | z={self.z_score:.2f} "
            f"p={self.p_value:.3g} | green {self.green_tokens}/{self.scored_tokens} "
            f"({self.green_fraction:.1%} vs {self.reason})"
        )


def _normal_sf(z: float) -> float:
    """One-sided tail of the standard normal. math.erfc is exact enough and stdlib."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# Below this many scored tokens the z-test has no power; report it rather than
# returning a confident-looking "not watermarked".
MIN_SCORED_TOKENS = 40


def detect_ids(
    token_ids: Sequence[int],
    config: WatermarkConfig,
    vocab_size: int,
    z_threshold: float = 4.0,
) -> DetectionResult:
    width = config.context_width
    ids = [int(t) for t in token_ids]
    green = 0
    scored = 0
    for i in range(width, len(ids)):
        mask = _green_mask_cpu(config.key, tuple(ids[i - width : i]), vocab_size, config.gamma)
        token = ids[i]
        if not 0 <= token < vocab_size:
            continue
        scored += 1
        green += int(bool(mask[token]))

    if scored == 0:
        return DetectionResult(config.identity, 0.0, 1.0, 0, 0, 0.0, False, "no scoreable tokens")

    expected = config.gamma * scored
    sigma = math.sqrt(scored * config.gamma * (1 - config.gamma))
    z = (green - expected) / sigma if sigma > 0 else 0.0
    p = _normal_sf(z)
    enough = scored >= MIN_SCORED_TOKENS
    reason = (
        f"{config.gamma:.0%} by chance"
        if enough
        else f"{config.gamma:.0%} by chance; only {scored} tokens, need {MIN_SCORED_TOKENS} to decide"
    )
    return DetectionResult(
        identity=config.identity,
        z_score=z,
        p_value=p,
        green_tokens=green,
        scored_tokens=scored,
        green_fraction=green / scored,
        watermarked=bool(enough and z >= z_threshold),
        reason=reason,
    )


def detect_text(
    text: str,
    tokenizer,
    config: WatermarkConfig,
    z_threshold: float = 4.0,
) -> DetectionResult:
    """Detect in raw text.

    Re-encoding is not guaranteed to reproduce the exact ids that were generated —
    byte-level BPE is close but a merge boundary can shift at an edit. That costs a
    little z, which is why the threshold defaults to a conservative 4.0 rather than
    the 2.0-ish a matched-ids test would justify.
    """
    ids = tokenizer.encode(text).ids
    return detect_ids(ids, config, tokenizer.get_vocab_size(), z_threshold=z_threshold)


@dataclass
class CapacityReport:
    """How much watermark a given model can actually carry on given text."""

    influenceable_fraction: float   # positions where the bias could change the choice
    median_entropy: float           # nats; the room a mark has to live in
    expected_green_rate: float      # mean post-bias probability of landing green
    expected_z_per_sqrt_token: float
    tokens_for_decisive_z: Optional[int]
    scored_positions: int

    def verdict(self, gamma: float) -> str:
        if self.scored_positions < 16:
            return "too little text to estimate capacity"
        if self.expected_z_per_sqrt_token < 0.05:
            return (
                f"this model is effectively unwatermarkable on this text: only "
                f"{self.influenceable_fraction:.0%} of positions have enough entropy for the "
                f"bias to change anything (median {self.median_entropy:.3f} nats). A green-list "
                "watermark can only hide in a choice the model was not already certain about."
            )
        if self.tokens_for_decisive_z and self.tokens_for_decisive_z > 2000:
            return (
                f"low capacity: {self.influenceable_fraction:.0%} of positions are "
                f"influenceable, so it would take roughly {self.tokens_for_decisive_z:,} tokens "
                "to reach a decisive z. Raise delta, or expect only long documents to be "
                "attributable."
            )
        return (
            f"healthy: {self.influenceable_fraction:.0%} of positions are influenceable, and "
            f"about {self.tokens_for_decisive_z:,} tokens should reach a decisive z."
        )


@torch.no_grad()
def watermark_capacity(
    model,
    tokenizer,
    text: str,
    config: WatermarkConfig,
    device=None,
    temperature: float = 0.8,
) -> CapacityReport:
    """Estimate how detectable a mark can be, given how certain this model is.

    A green-list watermark works by nudging a choice. Where the model is already
    certain — the second half of a common word, the closing bracket of a pair, the
    next token of a memorized template — there is no choice to nudge and the bias
    changes nothing. Such text carries no mark however large delta is, and nothing in
    the generated output reveals that. This is the single most common way a
    watermarking deployment silently does nothing.

    Rather than a heuristic, this computes the exact quantity the z-test depends on:
    the post-bias probability of landing on a green token at every position. Summing
    those gives the expected green count, and therefore the expected z, for text of
    any length.
    """
    import torch.nn.functional as F

    device = device or next(model.parameters()).device
    ids = tokenizer.encode(text).ids[: model.config.block_size]
    if len(ids) < 4:
        return CapacityReport(0.0, 0.0, config.gamma, 0.0, None, 0)

    logits, _ = model(torch.tensor([ids[:-1]], dtype=torch.long, device=device))
    scaled = logits[0].float() / max(temperature, 1e-5)
    vocab = scaled.size(-1)

    width = config.context_width
    green_rates: List[float] = []
    entropies: List[float] = []
    influenceable = 0
    for i in range(width - 1, scaled.size(0)):
        prefix = tuple(ids[i - width + 1 : i + 1])
        mask = _green_mask_cpu(config.key, prefix, vocab, config.gamma).to(device)
        row = scaled[i]
        probs = F.softmax(row, dim=-1)
        entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))

        biased = F.softmax(row + config.delta * mask, dim=-1)
        green_rates.append(float(biased[mask].sum()))
        # Could the bias move the argmax? Only if a green token is within delta of
        # the current best.
        best = float(row.max())
        best_green = float(row[mask].max()) if bool(mask.any()) else -float("inf")
        influenceable += int(best - best_green < config.delta)

    n = len(green_rates)
    if n == 0:
        return CapacityReport(0.0, 0.0, config.gamma, 0.0, None, 0)
    mean_green = sum(green_rates) / n
    sigma = math.sqrt(config.gamma * (1 - config.gamma))
    per_sqrt = (mean_green - config.gamma) / sigma if sigma > 0 else 0.0
    needed = int((4.0 / per_sqrt) ** 2) if per_sqrt > 1e-6 else None

    entropies.sort()
    return CapacityReport(
        influenceable_fraction=influenceable / n,
        median_entropy=entropies[n // 2],
        expected_green_rate=mean_green,
        expected_z_per_sqrt_token=per_sqrt,
        tokens_for_decisive_z=needed,
        scored_positions=n,
    )


def detect_best(
    text: str,
    tokenizer,
    candidates: Sequence[WatermarkConfig],
    z_threshold: float = 4.0,
) -> List[DetectionResult]:
    """Score one text against several candidate identities, strongest first.

    This is the attribution path: the identity whose key actually generated the text
    produces a large z, every other identity produces noise around zero.
    """
    results = [detect_text(text, tokenizer, c, z_threshold=z_threshold) for c in candidates]
    return sorted(results, key=lambda r: r.z_score, reverse=True)


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------

def _ask(question: str, default: str) -> str:
    try:
        answer = input(f"{question} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def _ask_yes_no(question: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            answer = input(f"{question} [{suffix}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Please answer y or n.")


def prompt_for_watermark(
    default_identity: str,
    *,
    context: str,
    existing: Optional[WatermarkConfig] = None,
    interactive: bool = True,
    default_on: bool = True,
) -> Optional[WatermarkConfig]:
    """Ask whether to watermark, and as what. Returns None for "do not watermark".

    Called at the two moments where the answer is actually knowable: when a model is
    first trained, and when it is released. Asking at generation time would be
    useless — by then the person running the model is not necessarily the person who
    needs the attribution.
    """
    if not interactive:
        return existing

    print()
    print(f"  Watermarking — {context}")
    print("  A statistical watermark biases token choice so generated text can later be")
    print("  attributed to this model. It is invisible to a reader, costs a small amount")
    print("  of output quality, needs ~40 tokens of text to detect, and does not survive")
    print("  heavy paraphrasing.")
    if existing:
        print(f"  This model currently identifies as {existing.identity!r}.")

    if not _ask_yes_no("  Watermark this model's output?", default_on):
        if existing:
            print("  Watermark removed.")
        return None

    identity = _ask("  Identify the model as", existing.identity if existing else default_identity)
    strength = _ask("  Strength — light / normal / strong", "normal")
    gamma, delta = {
        "light": (0.25, 1.0),
        "normal": (0.25, 2.0),
        "strong": (0.5, 4.0),
    }.get(strength.strip().lower(), (0.25, 2.0))
    if strength.strip().lower() not in {"light", "normal", "strong"}:
        print("  Unrecognized strength; using normal.")

    config = WatermarkConfig(identity=identity, gamma=gamma, delta=delta)
    print(f"  Watermark on: {identity!r} (gamma {gamma}, delta {delta}).")
    print(f"  Detect it later with:  geocentric detect --identity {identity!r} --text \"...\"")
    print()
    return config


def resolve_watermark(
    args,
    default_identity: str,
    *,
    context: str,
    existing: Optional[WatermarkConfig] = None,
) -> Tuple[Optional[WatermarkConfig], bool]:
    """Turn CLI flags plus an optional prompt into a decision.

    Returns (config, explicitly_removed). The second value matters because "no
    watermark was chosen" and "the existing watermark should be deleted" are
    different instructions to the caller.
    """
    import sys

    if getattr(args, "no_watermark", False):
        return None, True
    identity = getattr(args, "watermark_identity", None)
    if identity:
        return WatermarkConfig(
            identity=identity,
            gamma=getattr(args, "watermark_gamma", None) or 0.25,
            delta=getattr(args, "watermark_delta", None) or 2.0,
        ), False
    if getattr(args, "watermark", False):
        # --watermark with no identity: fall back to the model name rather than
        # refusing, so the flag is usable in a script.
        return WatermarkConfig(identity=default_identity), False
    if getattr(args, "yes", False) or not sys.stdin.isatty():
        return existing, False
    chosen = prompt_for_watermark(default_identity, context=context, existing=existing)
    return chosen, (existing is not None and chosen is None)
