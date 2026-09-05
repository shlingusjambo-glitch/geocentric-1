from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from geocentric.checkpoint import (
    find_tokenizer_path,
    load_checkpoint,
    pretrained_checkpoint_name,
    save_checkpoint,
    sft_checkpoint_name,
)
from geocentric.data import PadCollate, SFTDataset
from geocentric.device import cleanup, enable_fast_math, resolve_dtype, runtime_check, select_device
from geocentric.tokenizer_train import load_tokenizer, token_id
from geocentric.trainer import (
    Throughput,
    build_optimizer,
    format_progress,
    lr_at_step,
    maybe_compile,
    set_lr,
)
from geocentric.training_metrics import initialize_training_metrics, update_training_metrics
from tqdm.auto import tqdm


def sft(
    model_dir: str,
    sft_data_path: str,
    output_dir: str | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    eval_ratio: float = 0.02,
    dtype_name: str = "auto",
    gradient_checkpointing: bool = False,
    modelver: str = "Geocentric",
    overwrite_output_dir: bool = False,
    log_every: int = 10,
    num_workers: Optional[int] = None,
    compile_mode: str = "auto",
    drop_overlong: bool = True,
) -> None:
    src = Path(model_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else src
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    dtype = resolve_dtype(device, dtype_name)
    enable_fast_math()
    runtime_check(device, dtype)

    if overwrite_output_dir:
        for old in out.glob("*_sft*.pt"):
            old.unlink(missing_ok=True)

    tokenizer = load_tokenizer(find_tokenizer_path(src, extra_dirs=[out]))
    pad_id = token_id(tokenizer, "<pad>")

    model = load_checkpoint(
        src, device=device, dtype=torch.float32,
        checkpoint_name=pretrained_checkpoint_name(modelver, best=True),
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    for block in model.blocks:
        block.gradient_checkpointing = gradient_checkpointing
    block_size = model.config.block_size

    dataset = SFTDataset(tokenizer, sft_data_path, block_size=block_size, drop_overlong=drop_overlong)
    eval_len = max(1, int(len(dataset) * eval_ratio)) if len(dataset) > 50 else 0
    if eval_len:
        train_ds, eval_ds = random_split(
            dataset, [len(dataset) - eval_len, eval_len],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds, eval_ds = dataset, None

    if num_workers is None:
        num_workers = max(2, min(8, (os.cpu_count() or 2) - 1)) if device.type == "cuda" else 0
    collate = PadCollate(pad_id)
    loader_kwargs = dict(
        batch_size=batch_size, collate_fn=collate, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs) if eval_ds else None

    steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = max(5, int(total_steps * warmup_ratio))

    n_params = model.num_params()
    print(f"SFT: {len(dataset):,} conversations | {total_steps:,} steps | lr {learning_rate:.1e}")
    # 2e-5 is a fine-tune rate for a fully-trained multi-billion-parameter model. On a
    # small model pretrained from scratch it barely moves the weights, which is why
    # instruction following never took hold and the model echoed prompts back.
    if learning_rate < 3e-5:
        print(
            f"WARNING: learning_rate={learning_rate:.1e} is very low for a from-scratch model. "
            "Instruction following may not take hold. 1e-4 is a reasonable default here."
        )

    optimizer = build_optimizer(model, learning_rate, weight_decay, device_type=device.type)
    active_model, compiled = maybe_compile(model, device, compile_mode)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "mps"} and dtype != torch.float32
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    initialize_training_metrics(
        out, phase="sft",
        config={
            "examples": len(dataset), "dropped_overlong": getattr(dataset, "dropped", 0),
            "epochs": epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "total_steps": total_steps,
            "block_size": block_size, "dtype": str(dtype), "compiled": compiled,
        },
    )

    ckpt_name = sft_checkpoint_name(modelver)
    best_name = sft_checkpoint_name(modelver, best=True)
    meter = Throughput(n_params, block_size, device, dtype)
    step = 0
    best_eval = float("inf")
    running_loss = 0.0
    micro_count = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        pbar = tqdm(total=total_steps, desc="sft", dynamic_ncols=True)
        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast:
                    _, loss = active_model(input_ids, labels=labels)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    micro_count = 0
                    continue

                scaled = loss / gradient_accumulation_steps
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                running_loss += float(loss.detach())
                micro_count += 1
                meter.add(input_ids.numel())
                if micro_count < gradient_accumulation_steps:
                    continue

                lr = lr_at_step(step, total_steps, learning_rate, warmup_steps, min_lr_ratio)
                set_lr(optimizer, lr)
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(grad_norm):
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                elif use_scaler:
                    scaler.update()

                optimizer.zero_grad(set_to_none=True)
                step += 1
                avg_loss = running_loss / max(1, micro_count)
                running_loss = 0.0
                micro_count = 0
                pbar.update(1)

                if step % log_every == 0:
                    tps, mfu = meter.read()
                    pbar.set_description(format_progress(step, total_steps, avg_loss, lr, tps, mfu))
                    update_training_metrics(out, {
                        "step": step, "epoch": epoch, "loss": avg_loss, "lr": lr,
                        "perplexity": math.exp(min(avg_loss, 20)),
                        "tokens_per_second": tps, "message": "Training.",
                    })
                    meter.reset()

            if eval_loader is not None:
                eval_loss = evaluate(active_model, eval_loader, device, autocast)
                print(f"\n  epoch {epoch} eval loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
                update_training_metrics(out, {"eval_loss": eval_loss, "message": f"Epoch {epoch} evaluated."})
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    save_checkpoint(model, out, step, name=best_name, optimizer=optimizer, extra={"stage": "sft"})
                    print("  new best SFT checkpoint saved.")
                meter.reset()
            save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"stage": "sft"})
        pbar.close()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving SFT checkpoint before exit...")
        save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"stage": "sft"})
        update_training_metrics(out, {"status": "stopped", "message": "Interrupted by user."})
        return
    finally:
        cleanup(device)

    if not (out / best_name).exists():
        save_checkpoint(model, out, step, name=best_name, optimizer=optimizer, extra={"stage": "sft"})
    update_training_metrics(out, {"status": "complete", "message": "SFT complete."})
    print(f"SFT complete after {step:,} steps. Saved to {out}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, autocast) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast:
            _, loss = model(input_ids, labels=labels)
        if loss is not None and torch.isfinite(loss):
            total += float(loss.detach())
            count += 1
    model.train()
    return total / max(1, count)
