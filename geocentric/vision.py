"""Computer vision: a from-scratch ViT tower bolted onto the language model.

The wiring is the one the field settled on (LLaVA, Chen et al. 2023): encode the
image into a grid of patch states, project that grid into the language model's
embedding space, and *substitute* the result for a run of placeholder `<|image|>`
tokens in the prompt. The decoder is not modified at all — from its point of view
those positions simply hold unusual embeddings, and every mechanism it already has
(RoPE, GQA, the KV cache, the chat template) keeps working unchanged.

The encoder is trained from scratch rather than loaded from CLIP, because this whole
project is about training from scratch, and because a frozen CLIP tower would be
larger than most models trained here. That is a real quality ceiling — a scratch
tower on a small caption set will not match CLIP — and it is the honest tradeoff for
having no pretrained dependency. Point `--vision_init` at a checkpoint if you have
a better tower.

Attention in the tower is bidirectional: a picture has no arrow of time, and causal
masking would leave the top-left patch unable to see anything else in the image.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from geocentric.model import Block, GPTConfig, RMSNorm

IMAGE_TOKEN = "<|image|>"
# Roughly the midpoint of the usual normalizations. Nothing here is initialized from
# a pretrained encoder, so there is no statistic to match — only a need to centre.
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    pool: int = 2       # average-pool the patch grid pool x pool before projecting
    dropout: float = 0.0
    image_token_id: int = -1

    @property
    def grid(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}")
        return self.image_size // self.patch_size

    @property
    def n_patches(self) -> int:
        return self.grid * self.grid

    @property
    def n_tokens(self) -> int:
        """Image tokens spent per image, after pooling.

        224/16 is a 14x14 grid, which is 196 tokens — a fifth of a 1024-token context
        for one picture. Pooling 2x2 brings it to 49, which is the difference between
        a caption fitting alongside the image and not.
        """
        if self.grid % self.pool:
            raise ValueError(f"patch grid {self.grid} must be divisible by pool {self.pool}")
        return (self.grid // self.pool) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["VisionConfig"]:
        if not data:
            return None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class VisionTower(nn.Module):
    """Patch embed -> bidirectional transformer -> pool -> project into the LM space."""

    def __init__(self, config: VisionConfig, lm_embd: int) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            3, config.n_embd, kernel_size=config.patch_size, stride=config.patch_size
        )
        # Learned absolute positions. The image is a fixed-size grid, so there is
        # nothing for RoPE's extrapolation to buy here.
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_patches, config.n_embd))
        nn.init.normal_(self.pos_embed, std=0.02)

        inner = GPTConfig(
            vocab_size=1, block_size=config.n_patches, n_layer=config.n_layer,
            n_head=config.n_head, n_kv_head=config.n_head, n_embd=config.n_embd,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([Block(inner, causal=False) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)

        # Two layers, not one. A single linear projection is enough when the tower is
        # a pretrained CLIP whose space is already semantic; against a tower learned
        # from scratch on a small caption set the extra nonlinearity measurably helps.
        self.projector = nn.Sequential(
            nn.Linear(config.n_embd, lm_embd),
            nn.GELU(),
            nn.Linear(lm_embd, lm_embd),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, n_tokens, lm_embd)."""
        cfg = self.config
        if images.dim() != 4:
            raise ValueError(f"images must be (N, 3, H, W), got {tuple(images.shape)}")
        x = self.patch_embed(images.to(self.patch_embed.weight.dtype))  # (N, D, g, g)
        n, d, g, _ = x.shape
        x = x.flatten(2).transpose(1, 2) + self.pos_embed  # (N, P, D)

        # The tower is bidirectional, so RoPE would only add a second, redundant
        # position signal on top of pos_embed. Feed it identity rotation instead.
        cos = torch.ones(cfg.n_patches, d // cfg.n_head // 2, device=x.device, dtype=torch.float32)
        sin = torch.zeros_like(cos)
        for block in self.blocks:
            x = block(x, cos, sin, None)
        x = self.ln_f(x)

        if cfg.pool > 1:
            x = x.transpose(1, 2).reshape(n, d, g, g)
            x = torch.nn.functional.avg_pool2d(x, cfg.pool)
            x = x.flatten(2).transpose(1, 2)
        return self.projector(x)

    def splice(self, embeds: torch.Tensor, input_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Replace every `<|image|>` embedding with the matching patch state.

        images is (N, 3, H, W) with N images across the whole batch, in the order the
        placeholders appear when the batch is read row by row.
        """
        cfg = self.config
        if cfg.image_token_id < 0:
            raise ValueError("VisionConfig.image_token_id is unset — call attach_vision().")
        if images.dim() == 5:
            images = images.flatten(0, 1)
        mask = input_ids == cfg.image_token_id
        wanted = int(mask.sum())
        have = images.size(0) * cfg.n_tokens
        if wanted != have:
            raise ValueError(
                f"{wanted} <|image|> placeholders in the batch but {images.size(0)} images "
                f"x {cfg.n_tokens} tokens = {have}. Build prompts with image_placeholder()."
            )
        if wanted == 0:
            return embeds
        feats = self.encode(images).reshape(-1, embeds.size(-1))
        return embeds.masked_scatter(mask.unsqueeze(-1), feats.to(embeds.dtype))


def image_placeholder(config: VisionConfig) -> str:
    """The run of placeholder tokens one image occupies in a prompt."""
    return IMAGE_TOKEN * config.n_tokens


def attach_vision(model, tokenizer, config: Optional[VisionConfig] = None):
    """Give a language model a vision tower, growing the tokenizer if it needs to.

    Checkpoints trained before this existed have no `<|image|>` token. Rather than
    forcing a retrain, the token is appended and the embedding matrix is grown by one
    row, seeded with the mean of the existing embeddings so it starts life as a
    perfectly average token instead of an outlier the model has to unlearn. The row
    is immediately overwritten by the splice at every position that matters, so its
    value only affects the (masked) loss bookkeeping — but a NaN-adjacent init here
    would still poison the tied lm_head.
    """
    config = config or VisionConfig()
    token_id = tokenizer.token_to_id(IMAGE_TOKEN)
    if token_id is None:
        from tokenizers import AddedToken

        tokenizer.add_special_tokens([AddedToken(IMAGE_TOKEN, special=True, normalized=False)])
        token_id = tokenizer.token_to_id(IMAGE_TOKEN)
        _grow_embeddings(model, tokenizer.get_vocab_size())
    elif tokenizer.get_vocab_size() > model.config.vocab_size:
        _grow_embeddings(model, tokenizer.get_vocab_size())
    config.image_token_id = int(token_id)

    tower = VisionTower(config, model.config.n_embd)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # nn.Module.__setattr__ registers this as a submodule, so the tower rides along
    # in state_dict() under "vision.*" with no extra bookkeeping.
    model.vision = tower.to(device=device, dtype=dtype)
    model.config.vision = config.to_dict()
    return model


def _grow_embeddings(model, vocab_size: int) -> None:
    old = model.token_embedding
    if vocab_size <= old.num_embeddings:
        return
    grown = nn.Embedding(vocab_size, old.embedding_dim).to(old.weight.device, old.weight.dtype)
    with torch.no_grad():
        grown.weight[: old.num_embeddings] = old.weight
        grown.weight[old.num_embeddings :] = old.weight.mean(dim=0, keepdim=True)
    model.token_embedding = grown
    model.lm_head = nn.Linear(old.embedding_dim, vocab_size, bias=False).to(
        old.weight.device, old.weight.dtype
    )
    model.lm_head.weight = model.token_embedding.weight  # stay tied
    model.config.vocab_size = vocab_size


# ---------------------------------------------------------------------------
# Images on disk
# ---------------------------------------------------------------------------

def load_image(path: str | Path, size: int = 224) -> torch.Tensor:
    """Read one image as a normalized (3, size, size) float tensor."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError(
            "Vision needs Pillow. Install it with: pip install 'geocentric[vision]'"
        ) from exc
    import numpy as np

    with Image.open(path) as img:
        # Centre-crop to square first: a plain resize squashes a 16:9 photo into a
        # square and the model learns the distortion along with the content.
        img = img.convert("RGB")
        short = min(img.size)
        left = (img.width - short) // 2
        top = (img.height - short) // 2
        img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.BICUBIC)
        array = np.asarray(img, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - PIXEL_MEAN) / PIXEL_STD


def save_vision_config(directory: str | Path, config: VisionConfig) -> Path:
    path = Path(directory) / "vision_config.json"
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return path
