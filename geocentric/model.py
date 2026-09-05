from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    model_name: str = "Geocentric"
    gradient_checkpointing: bool = False
    # Sidecar dicts rather than nested dataclasses so a config round-trips through
    # plain JSON and an older checkpoint that lacks them still loads.
    watermark: Optional[dict] = None
    vision: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even attention head dimension")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GPTConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab_size=data.get("vocab_size", 0),
            block_size=data.get("block_size", data.get("max_position_embeddings", 1024)),
            n_layer=data.get("n_layer", data.get("num_hidden_layers", 12)),
            n_head=data.get("n_head", data.get("num_attention_heads", 12)),
            n_kv_head=data.get("n_kv_head", data.get("num_key_value_heads")),
            n_embd=data.get("n_embd", data.get("hidden_size", 768)),
            dropout=data.get("dropout", 0.0),
            bias=data.get("bias", False),
            rope_theta=data.get("rope_theta", 10000.0),
            norm_eps=data.get("norm_eps", 1e-5),
            model_name=data.get("model_name", data.get("model_type", "Geocentric")),
            gradient_checkpointing=data.get("gradient_checkpointing", False),
            watermark=data.get("watermark"),
            vision=data.get("vision"),
        )


class RMSNorm(nn.Module):
    """RMSNorm — no mean subtraction, no bias. Cheaper than LayerNorm and empirically
    equal or better for decoder-only LMs (used by Llama, Mistral, Gemma)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for stability under bf16 autocast, then cast back.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    head_dim: int, max_len: int, theta: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D/2). Rotary position embedding."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :].float()
    sin = sin[None, None, :, :].float()
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


class KVCache:
    """Per-layer key/value cache so generation is O(1) per token instead of O(T)."""

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.size(2)


class CausalSelfAttention(nn.Module):
    """Causal attention with rotary embeddings and optional grouped-query attention.

    GQA (n_kv_head < n_head) shrinks the KV cache and the KV projections, which is
    where most of the memory and bandwidth cost of generation lives.
    """

    def __init__(self, config: GPTConfig, causal: bool = True) -> None:
        super().__init__()
        # An image is not a sequence in time: a patch may attend to every other patch.
        # Forcing causal masking on the vision tower would make the top-left patch
        # blind to the rest of the picture.
        self.causal = causal
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        b, t, c = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # is_causal is only correct when q and k have equal length. During cached
        # decoding a single query attends to the whole prefix, which is already causal.
        is_causal = self.causal and (cache is None or t > 1)
        mask = None
        if self.causal and cache is not None and k.size(2) > t and t > 1:
            # SDPA's rectangular causal mask is upper-left aligned. A cached
            # multi-token continuation needs the lower-right prefix offset.
            prefix = k.size(2) - t
            mask = torch.arange(k.size(2), device=x.device)[None, :] <= (
                prefix + torch.arange(t, device=x.device)[:, None]
            )
            is_causal = False
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward. Consistently beats GELU at matched parameter count,
    which is why every modern open LM uses it."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # 8/3 * n_embd keeps parameter count equal to a 4x GELU MLP despite the
        # third projection, then round to a multiple of 128 for tensor-core alignment.
        hidden = int(8 * config.n_embd / 3)
        hidden = 128 * ((hidden + 127) // 128)
        self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, causal: bool = True) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, causal=causal)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLU(config)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            import torch.utils.checkpoint

            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, cos, sin, cache, use_reentrant=False
            )
        return self._forward_impl(x, cos, sin, cache)

    def _forward_impl(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache],
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin, cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GeocentricGPT(nn.Module):
    """Decoder-only causal language model trained from random init.

    Architecture: pre-norm residual blocks, RMSNorm, rotary position embeddings,
    grouped-query attention, SwiGLU feed-forward, tied input/output embeddings.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scaled init on residual output projections. Without this the residual
        # stream variance grows with depth and deep models train unstably or waste
        # the first several thousand steps recovering. (GPT-2 paper, section 2.3.)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=residual_std)

        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._rope_key: Optional[Tuple[int, str, torch.dtype]] = None

        # Elastic depth (see geocentric/epicycle.py). None means "every block".
        # Blocks past this index are skipped in the forward pass, so they receive no
        # gradient and — because zero_grad(set_to_none=True) leaves their .grad at
        # None — AdamW skips them entirely, weight decay included.
        self.active_layers: Optional[int] = None
        # Runtime-only setting; checkpoint architecture and default API stay intact.
        self.loss_chunk_size = 0
        # Optional vision tower, attached by geocentric.vision.attach_vision().
        self.vision = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.n_embd // self.config.n_head
        key = (self.config.block_size, str(device), dtype)
        if self._rope_key != key:
            self._rope_cache = build_rope_cache(
                head_dim, self.config.block_size, self.config.rope_theta, device, dtype
            )
            self._rope_key = key
        assert self._rope_cache is not None
        return self._rope_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        caches: Optional[List[KVCache]] = None,
        position_offset: int = 0,
        images: Optional[torch.Tensor] = None,
        loss_reduction: str = "mean",
        return_logits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        b, t = input_ids.shape
        if position_offset + t > self.config.block_size:
            raise ValueError(
                f"Sequence position {position_offset + t} exceeds block size {self.config.block_size}"
            )

        x = self.token_embedding(input_ids)
        if images is not None:
            if self.vision is None:
                raise ValueError("images= was passed but this checkpoint has no vision tower.")
            x = self.vision.splice(x, input_ids, images)
        x = self.dropout(x)

        cos_all, sin_all = self._rope(input_ids.device, torch.float32)
        cos = cos_all[position_offset : position_offset + t]
        sin = sin_all[position_offset : position_offset + t]

        depth = len(self.blocks) if self.active_layers is None else max(1, self.active_layers)
        for i, block in enumerate(self.blocks[:depth]):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        x = self.ln_f(x)

        loss = None
        if labels is not None and not return_logits and self.loss_chunk_size > 0:
            from geocentric.streaming_loss import linear_cross_entropy

            logits = None
            loss = linear_cross_entropy(x, self.lm_head.weight, labels,
                                        self.loss_chunk_size, loss_reduction)
        elif labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
                # "none" returns one loss per position, which is what lets EQUANT
                # choose which tokens are worth a gradient. Shape (B*T,).
                reduction=loss_reduction,
            )
            if not return_logits:
                logits = None
        else:
            # Inference only needs the last position's logits. Projecting the whole
            # sequence through a 32k-wide head is pure waste during generation.
            logits = self.lm_head(x[:, -1:, :]) if caches is not None else self.lm_head(x)
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        eos_id: Optional[int] = None,
        repetition_penalty: float = 1.1,
        repetition_window: int = 128,
        logits_processor=None,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        caches = [KVCache() for _ in self.blocks]
        prompt_len = input_ids.size(1)
        if prompt_len > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
            prompt_len = input_ids.size(1)

        generated = input_ids
        cur = input_ids
        offset = 0

        for _ in range(max_new_tokens):
            # Stop before the sequence itself would exceed the context, not merely
            # before the next forward pass would.
            if generated.size(1) >= self.config.block_size:
                break
            if offset + cur.size(1) > self.config.block_size:
                break
            # Images only enter on the prefill pass; afterwards their patch states
            # live in the KV cache and re-splicing them would double-count.
            logits, _ = self(cur, caches=caches, position_offset=offset,
                             images=images if offset == 0 else None)
            offset += cur.size(1)
            logits = logits[:, -1, :].float()

            # Penalize only the recent window. Penalizing the entire prompt makes the
            # model avoid the user's own words, which reads as evasive and off-topic.
            if repetition_penalty != 1.0 and repetition_window > 0:
                recent = generated[:, -repetition_window:]
                score = torch.gather(logits, 1, recent)
                score = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty
                )
                logits.scatter_(1, recent, score)

            if temperature > 0:
                logits = logits / max(temperature, 1e-5)
            # Applied after temperature so the watermark bias means the same thing
            # at temp 0.3 and temp 1.2 — dividing it would silently weaken the mark.
            if logits_processor is not None:
                logits = logits_processor(logits, generated)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                if min_p > 0.0:
                    threshold = min_p * probs.max(dim=-1, keepdim=True).values
                    probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    keep = cumulative - sorted_probs < top_p
                    keep[:, 0] = True
                    sorted_probs = sorted_probs * keep
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                total = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(total > 0, probs / total, torch.ones_like(probs) / probs.size(-1))
                next_id = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_id], dim=1)
            cur = next_id
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
