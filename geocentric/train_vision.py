"""Stage the language model into a multimodal one on image/text pairs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from geocentric.chat import messages_from_record, render_chat
from geocentric.checkpoint import find_tokenizer_path, load_checkpoint, save_checkpoint
from geocentric.device import cleanup, enable_fast_math, resolve_dtype, runtime_check, select_device
from geocentric.tokenizer_train import load_tokenizer, token_id
from geocentric.trainer import Throughput, format_progress, lr_at_step
from geocentric.training_metrics import initialize_training_metrics, update_training_metrics
from geocentric.vision import (
    VisionConfig,
    attach_vision,
    image_placeholder,
    load_image,
    save_vision_config,
)
from geocentric.watermark import WatermarkConfig

DEFAULT_IMAGE_PROMPTS = [
    "Describe this image.",
    "What is in this picture?",
    "Caption this image.",
]


def vision_checkpoint_name(modelver: str = "Geocentric", *, best: bool = False) -> str:
    from geocentric.checkpoint import modelver_to_filename

    return f"{modelver_to_filename(modelver)}_vision{'_best' if best else ''}.pt"


class VisionSFTDataset(Dataset):
    """Image/text conversations, with everything but the assistant's turns masked.

    Accepts the shapes people actually have on disk:
        {"image": "cat.jpg", "caption": "A cat on a sofa."}
        {"image": "cat.jpg", "instruction": "What animal is this?", "output": "A cat."}
        {"image": "cat.jpg", "messages": [{"role": "user", ...}, ...]}
    plus "images": [...] for several pictures in one conversation. Relative paths
    resolve against the data file's own directory first, which is how every caption
    dataset is actually laid out, then against the working directory.
    """

    def __init__(
        self,
        tokenizer,
        path: str | Path,
        vision: VisionConfig,
        block_size: int,
        image_root: Optional[str | Path] = None,
        drop_overlong: bool = True,
    ) -> None:
        self.vision = vision
        self.image_size = vision.image_size
        data_path = Path(path).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Vision data not found: {data_path}")
        self.roots = [Path(image_root).expanduser().resolve()] if image_root else []
        self.roots += [data_path.parent, Path.cwd()]

        placeholder = image_placeholder(vision)
        rows: List[Tuple[str, List[Tuple[int, int]], List[Path]]] = []
        self.missing = 0
        self.dropped = 0

        for record in self._read_rows(data_path):
            images = self._image_paths(record)
            if not images:
                self.missing += 1
                continue
            messages = self._messages(record)
            if not messages:
                continue
            # The placeholders go at the head of the first user turn: the model must
            # see the picture before the question about it, not after.
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = f"{placeholder * len(images)}\n{msg['content']}"
                    break
            else:
                continue
            text, spans = render_chat(messages)
            rows.append((text, spans, images))

        if not rows:
            raise ValueError(
                f"No usable image/text pairs in {data_path} "
                f"({self.missing:,} records had a missing or unreadable image path)."
            )

        self.examples: List[Dict[str, object]] = []
        encodings = tokenizer.encode_batch([t for t, _, _ in rows])
        image_id = vision.image_token_id
        for (text, spans, images), enc in zip(rows, encodings):
            ids = enc.ids
            if len(ids) < 2 or (drop_overlong and len(ids) > block_size):
                self.dropped += 1
                continue
            ids = ids[:block_size]
            n_placeholders = sum(1 for i in ids if i == image_id)
            if n_placeholders != len(images) * vision.n_tokens:
                # Truncation ate part of an image, or the tokenizer split the
                # placeholder. Either way the splice would raise mid-epoch.
                self.dropped += 1
                continue

            labels = [-100] * len(ids)
            for i, (start, end) in enumerate(enc.offsets[: len(ids)]):
                if end <= start:
                    continue
                for span_start, span_end in spans:
                    if start >= span_start and end <= span_end:
                        labels[i] = ids[i]
                        break
            if all(label == -100 for label in labels[1:]):
                continue
            self.examples.append({
                "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                "labels": torch.tensor(labels[1:], dtype=torch.long),
                "image_paths": images,
            })

        if not self.examples:
            raise ValueError(f"No usable examples survived preprocessing in {data_path}")
        if self.missing or self.dropped:
            print(f"Vision data: {len(self.examples):,} usable | {self.missing:,} missing images "
                  f"| {self.dropped:,} dropped (too long for block_size={block_size})")

    def _messages(self, record: Mapping[str, object]) -> List[dict]:
        messages = messages_from_record(record)
        if messages:
            return messages
        caption = str(record.get("caption", record.get("text", ""))).strip()
        if not caption:
            return []
        # Rotate the prompt so the model learns to answer the question rather than
        # to emit a caption whenever it sees a picture.
        prompt = DEFAULT_IMAGE_PROMPTS[len(caption) % len(DEFAULT_IMAGE_PROMPTS)]
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": caption}]

    def _image_paths(self, record: Mapping[str, object]) -> List[Path]:
        raw = record.get("images") or record.get("image")
        if not raw:
            return []
        candidates = raw if isinstance(raw, list) else [raw]
        found: List[Path] = []
        for name in candidates:
            for root in self.roots:
                path = (root / str(name)).resolve()
                if path.exists():
                    found.append(path)
                    break
            else:
                direct = Path(str(name)).expanduser()
                if direct.exists():
                    found.append(direct.resolve())
                else:
                    return []
        return found

    @staticmethod
    def _read_rows(path: Path) -> Iterator[Mapping[str, object]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield from (payload if isinstance(payload, list) else [payload])
            return
        raise ValueError("Vision data must be .json or .jsonl")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.examples[idx]
        # Decoded in the worker, not the constructor: a caption set is tens of
        # thousands of JPEGs and they will not fit in RAM as tensors.
        images = torch.stack([load_image(p, self.image_size) for p in item["image_paths"]])
        return {"input_ids": item["input_ids"], "labels": item["labels"], "images": images}


class VisionCollate:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        width = max(int(x["input_ids"].numel()) for x in batch)
        ids, labels = [], []
        for item in batch:
            pad = width - int(item["input_ids"].numel())
            ids.append(torch.cat([item["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(ids),
            "labels": torch.stack(labels),
            # Flat across the batch, in placeholder order — exactly what splice wants.
            "images": torch.cat([x["images"] for x in batch]),
        }


def train_vision(
    model_dir: str,
    vision_data_path: str,
    output_dir: Optional[str] = None,
    image_root: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    projector_lr_multiplier: float = 5.0,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    eval_ratio: float = 0.02,
    image_size: int = 224,
    patch_size: int = 16,
    vision_layers: int = 6,
    vision_width: int = 384,
    vision_pool: int = 2,
    freeze_lm: bool = False,
    dtype_name: str = "auto",
    modelver: str = "Geocentric",
    log_every: int = 10,
    num_workers: Optional[int] = None,
    watermark: Optional[WatermarkConfig] = None,
    loss_chunk_size: int = 0,
) -> None:
    if gradient_accumulation_steps < 1 or loss_chunk_size < 0:
        raise ValueError("accumulation must be positive and loss_chunk_size nonnegative")
    src = Path(model_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else src
    out.mkdir(parents=True, exist_ok=True)
    if Path(vision_data_path).expanduser().resolve() in {
        out / "vision_config.json", out / "tokenizer.json", out / "config.json"
    }:
        raise ValueError("Vision data path conflicts with a training output filename")

    device = select_device()
    dtype = resolve_dtype(device, dtype_name)
    enable_fast_math()
    runtime_check(device, dtype)

    tokenizer = load_tokenizer(find_tokenizer_path(src, extra_dirs=[out]))
    model = load_checkpoint(src, device=device, dtype=torch.float32)
    model.loss_chunk_size = loss_chunk_size
    if watermark is not None:
        model.config.watermark = watermark.to_dict()
        watermark.save(out)

    vision = VisionConfig(
        image_size=image_size, patch_size=patch_size, n_embd=vision_width,
        n_layer=vision_layers, n_head=max(1, vision_width // 64), pool=vision_pool,
    )
    attach_vision(model, tokenizer, vision)
    vision = VisionConfig.from_dict(model.config.vision)
    tokenizer.save(str(out / "tokenizer.json"))
    save_vision_config(out, vision)
    print(f"Vision tower: {vision.n_layer}L x {vision.n_embd}d | {vision.image_size}px "
          f"/ {vision.patch_size} patches -> {vision.n_tokens} image tokens per image")

    dataset = VisionSFTDataset(tokenizer, vision_data_path, vision, model.config.block_size,
                               image_root=image_root)
    eval_len = max(1, int(len(dataset) * eval_ratio)) if len(dataset) > 50 else 0
    if eval_len:
        train_ds, eval_ds = random_split(
            dataset, [len(dataset) - eval_len, eval_len],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds, eval_ds = dataset, None

    if num_workers is None:
        num_workers = max(2, min(6, (os.cpu_count() or 2) - 1)) if device.type != "cpu" else 0
    collate = VisionCollate(token_id(tokenizer, "<pad>"))
    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate, num_workers=num_workers,
                         pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs) if eval_ds else None

    if freeze_lm:
        # Stage 1 of the usual two-stage recipe: align the projector against a fixed
        # language model before letting gradients into the decoder. Worth doing when
        # the caption set is small enough that full training would just overwrite the
        # text ability you spent a week on.
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("vision")
        print("Language model frozen: training the vision tower and projector only.")

    tower = model.vision
    projector_params = list(tower.projector.parameters())
    projector_ids = {id(p) for p in projector_params}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in projector_ids]
    # The projector is the only randomly-initialized bridge between two spaces that
    # do not yet agree; at the decoder's learning rate it converges far too slowly
    # and the decoder drifts to meet it instead, which costs text quality.
    optimizer = torch.optim.AdamW(
        [{"params": rest, "lr": learning_rate, "weight_decay": weight_decay},
         {"params": projector_params, "lr": learning_rate * projector_lr_multiplier,
          "weight_decay": 0.0}],
        betas=(0.9, 0.95), eps=1e-8,
        fused=device.type == "cuda",
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    if not len(train_loader):
        raise ValueError("No image/text training windows remain after filtering")
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = max(5, int(total_steps * warmup_ratio))
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model._grad_scaler = scaler
    loss_normalizer = batch_size * model.config.block_size * gradient_accumulation_steps
    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "mps"} and dtype != torch.float32
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    initialize_training_metrics(out, phase="vision", config={
        "examples": len(dataset), "missing_images": dataset.missing, "dropped": dataset.dropped,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "total_steps": total_steps, "vision": vision.to_dict(), "freeze_lm": freeze_lm,
        "loss_chunk_size": loss_chunk_size, "use_scaler": use_scaler,
        "watermark_identity": (model.config.watermark or {}).get("identity"),
    })

    name = vision_checkpoint_name(modelver)
    best_name = vision_checkpoint_name(modelver, best=True)
    meter = Throughput(model.num_params(), model.config.block_size, device, dtype)
    step, best_eval, running, micro = 0, float("inf"), 0.0, 0
    window_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        pbar = tqdm(total=total_steps, desc="vision", dynamic_ncols=True)
        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            for batch_index, batch in enumerate(train_loader):
                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                images = batch["images"].to(device, non_blocking=True)

                with autocast:
                    _, loss = model(ids, labels=labels, images=images, return_logits=False,
                                    loss_reduction="sum")
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    running = 0.0
                    window_tokens = 0
                    continue
                scaler.scale(loss / loss_normalizer).backward()
                running += float(loss.detach())
                micro += 1
                window_tokens += int((labels != -100).sum())
                meter.add(ids.numel())
                if micro < gradient_accumulation_steps and batch_index + 1 < len(train_loader):
                    continue
                if window_tokens == 0:
                    optimizer.zero_grad(set_to_none=True)
                    running, micro = 0.0, 0
                    continue

                scale = lr_at_step(step, total_steps, 1.0, warmup_steps)
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * scale
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(loss_normalizer / window_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                avg = running / window_tokens
                running, micro = 0.0, 0
                window_tokens = 0
                pbar.update(1)

                if step % log_every == 0:
                    tps, mfu = meter.read()
                    pbar.set_description(
                        format_progress(step, total_steps, avg, base_lrs[0] * scale, tps, mfu)
                    )
                    update_training_metrics(out, {
                        "step": step, "epoch": epoch, "loss": avg,
                        "perplexity": math.exp(min(avg, 20)), "tokens_per_second": tps,
                        "message": "Training.",
                    })
                    meter.reset()

            if eval_loader is not None:
                eval_loss = evaluate(model, eval_loader, device, autocast)
                print(f"\n  epoch {epoch} eval loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
                update_training_metrics(out, {"eval_loss": eval_loss, "message": f"Epoch {epoch} evaluated."})
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    save_checkpoint(model, out, step, name=best_name, optimizer=optimizer,
                                    extra={"stage": "vision"})
                meter.reset()
            save_checkpoint(model, out, step, name=name, optimizer=optimizer, extra={"stage": "vision"})
        pbar.close()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving vision checkpoint before exit...")
        save_checkpoint(model, out, step, name=name, optimizer=optimizer, extra={"stage": "vision"})
        update_training_metrics(out, {"status": "stopped", "message": "Interrupted by user."})
        return
    finally:
        cleanup(device)

    if not (out / best_name).exists():
        save_checkpoint(model, out, step, name=best_name, optimizer=optimizer, extra={"stage": "vision"})
    update_training_metrics(out, {"status": "complete", "step": step,
                                  "message": "Vision training complete."})
    print(f"Vision training complete after {step:,} steps. Saved to {out}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, autocast) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        with autocast:
            _, loss = model(
                batch["input_ids"].to(device), labels=batch["labels"].to(device),
                images=batch["images"].to(device),
                return_logits=False, loss_reduction="sum",
            )
        if loss is not None and torch.isfinite(loss):
            total += float(loss.detach())
            count += int((batch["labels"] != -100).sum())
    model.train()
    return total / max(1, count)
