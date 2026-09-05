from __future__ import annotations

import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from geocentric.checkpoint import (
    load_checkpoint,
    load_optimizer_state,
    pretrained_checkpoint_name,
    save_checkpoint,
)
from geocentric.data import PackedDataset, corpus_paths, iter_documents, prepare_corpus
from geocentric.device import cleanup, enable_fast_math, peak_memory_gb, resolve_dtype, runtime_check, select_device
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.param_compiler import recommended_tokens
from geocentric.tokenizer_train import DEFAULT_VOCAB_SIZE, load_tokenizer, train_byte_bpe_tokenizer
from geocentric.trainer import (
    Throughput,
    build_optimizer,
    find_batch_size,
    format_progress,
    lr_at_step,
    maybe_compile,
    set_lr,
)
from geocentric.training_metrics import initialize_training_metrics, update_training_metrics


def pretrain(
    data_path: str,
    output_dir: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    block_size: int = 1024,
    n_layer: int = 12,
    n_head: int = 12,
    n_kv_head: Optional[int] = None,
    n_embd: int = 768,
    dropout: float = 0.0,
    epochs: int = 1,
    max_steps: int = 0,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 6e-4,
    warmup_ratio: float = 0.01,
    min_lr_ratio: float = 0.1,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    val_fraction: float = 0.005,
    doc_sep: Optional[str] = None,
    dtype_name: str = "auto",
    tokenizer_path: Optional[str] = None,
    gradient_checkpointing: bool = False,
    modelver: str = "Geocentric",
    overwrite_output_dir: bool = False,
    log_every: int = 20,
    eval_every: int = 500,
    save_every: int = 1000,
    num_workers: Optional[int] = None,
    compile_mode: str = "auto",
    resume: bool = True,
    force_reprepare: bool = False,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    dtype = resolve_dtype(device, dtype_name)
    enable_fast_math()
    runtime_check(device, dtype)

    if overwrite_output_dir:
        for pattern in ("*_pretrained*.pt", "model.pt"):
            for old in out.glob(pattern):
                old.unlink(missing_ok=True)
        print("Overwrite requested: removed existing pretraining checkpoints.")

    # ---- tokenizer -------------------------------------------------------
    tok_out = out / "tokenizer.json"
    if tokenizer_path:
        import shutil

        shutil.copyfile(tokenizer_path, tok_out)
        tokenizer = load_tokenizer(tok_out)
    elif tok_out.exists():
        tokenizer = load_tokenizer(tok_out)
    else:
        print(f"Training byte-level BPE tokenizer (vocab {vocab_size:,}) from scratch...")
        tokenizer = train_byte_bpe_tokenizer(iter_documents(data_path, doc_sep), tok_out, vocab_size=vocab_size)

    # ---- corpus ----------------------------------------------------------
    corpus_dir = out / "corpus"
    train_bin, train_meta = corpus_paths(corpus_dir, "train")
    val_bin, val_meta = corpus_paths(corpus_dir, "val")
    if force_reprepare or not train_bin.exists():
        print("Tokenizing corpus to a binary token stream (one-time cost, reused on later runs)...")
        prepare_corpus(tokenizer, data_path, corpus_dir, val_fraction=val_fraction, doc_sep=doc_sep)

    meta = json.loads(train_meta.read_text(encoding="utf-8"))
    train_ds = PackedDataset(train_bin, block_size=block_size, dtype=meta["dtype"])
    eval_ds = (
        PackedDataset(val_bin, block_size=block_size, dtype=meta["dtype"])
        if val_bin.exists() and val_bin.stat().st_size > block_size * 4
        else None
    )

    # ---- model -----------------------------------------------------------
    config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        dropout=dropout,
        gradient_checkpointing=gradient_checkpointing,
        model_name=modelver,
    )

    ckpt_name = pretrained_checkpoint_name(modelver)
    best_name = pretrained_checkpoint_name(modelver, best=True)
    start_step = 0
    if resume and (out / ckpt_name).exists():
        model = load_checkpoint(out, device=device, dtype=torch.float32, checkpoint_name=ckpt_name)
        config = model.config
    else:
        print("Initializing model from random weights.")
        model = GeocentricGPT(config).to(device)

    # Master weights stay float32; autocast handles the low-precision compute. Casting
    # the parameters themselves to bf16 discards mantissa bits on every update and is
    # a large part of why small from-scratch models plateau early.
    model = model.to(device=device, dtype=torch.float32)

    n_params = model.num_params()
    budget = recommended_tokens(n_params)
    total_corpus_tokens = meta["tokens"]
    print(f"Model parameters: {n_params:,}")
    print(f"Corpus: {total_corpus_tokens:,} training tokens across {len(train_ds):,} windows of {block_size}")
    print(f"Compute-optimal budget for this size is about {budget:,} tokens ({budget / 1e9:.1f}B).")
    if total_corpus_tokens * max(1, epochs) < budget * 0.5:
        seen = total_corpus_tokens * max(1, epochs)
        print(
            f"WARNING: this run will see ~{seen:,} tokens, {budget / max(1, seen):.1f}x below the "
            f"compute-optimal budget. Expect fluent but shallow, off-topic output. Add more data "
            f"or raise --epochs before blaming the architecture."
        )

    # ---- batch sizing ----------------------------------------------------
    if batch_size <= 0:
        batch_size = find_batch_size(
            model, block_size, device, dtype,
            learning_rate=learning_rate, weight_decay=weight_decay,
        )
        if gradient_accumulation_steps <= 0:
            # Aim for roughly half a million tokens per optimizer step, the range
            # small models train most stably in.
            gradient_accumulation_steps = max(1, round(500_000 / (batch_size * block_size)))
            print(
                f"Gradient accumulation: {gradient_accumulation_steps} "
                f"({batch_size * gradient_accumulation_steps * block_size:,} tokens per step)"
            )
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    # ---- data loading ----------------------------------------------------
    if num_workers is None:
        num_workers = max(2, min(8, (os.cpu_count() or 2) - 1)) if device.type == "cuda" else 0
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    # drop_last stays off for eval: a small validation split can hold fewer windows
    # than one batch, and dropping it made evaluate() silently report a loss of 0.
    eval_kwargs = {**loader_kwargs, "drop_last": False}
    eval_loader = DataLoader(eval_ds, shuffle=False, **eval_kwargs) if eval_ds else None

    steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * max(1, epochs)
    warmup_steps = max(10, int(total_steps * warmup_ratio))

    optimizer = build_optimizer(model, learning_rate, weight_decay, device_type=device.type)
    if resume and (out / ckpt_name).exists():
        start_step = load_optimizer_state(out, ckpt_name, optimizer)
        if start_step:
            print(f"Resuming at step {start_step:,} of {total_steps:,}.")

    active_model, compiled = maybe_compile(model, device, compile_mode)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "mps"} and dtype != torch.float32
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    tokens_per_step = batch_size * gradient_accumulation_steps * block_size
    print(
        f"Schedule: {total_steps:,} optimizer steps x {tokens_per_step:,} tokens/step "
        f"= {total_steps * tokens_per_step:,} tokens | warmup {warmup_steps:,} steps"
    )

    initialize_training_metrics(
        out,
        phase="pretraining",
        config={
            "vocab_size": config.vocab_size, "block_size": block_size, "n_layer": config.n_layer,
            "n_head": config.n_head, "n_kv_head": config.n_kv_head, "n_embd": config.n_embd,
            "params": n_params, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "total_steps": total_steps,
            "tokens_per_step": tokens_per_step, "dtype": str(dtype), "compiled": compiled,
            "corpus_tokens": total_corpus_tokens, "recommended_tokens": budget,
            # Stored so tooling can show the exact line that resumes this run.
            "command": " ".join(shlex.quote(a) for a in [sys.executable, "-m", "geocentric.cli", *sys.argv[1:]]),
        },
    )

    # Publish the pid so monitoring tools identify this run unambiguously.
    pid_file = out / "trainer.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    meter = Throughput(n_params, block_size, device, dtype)
    step = start_step
    best_eval = float("inf")
    running_loss = 0.0
    micro_count = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        pbar = tqdm(total=total_steps, initial=step, desc="pretrain", dynamic_ncols=True)
        done = False
        while not done:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast:
                    _, loss = active_model(input_ids, labels=labels)
                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss at step {step}; skipping microbatch.")
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
                else:
                    print(f"WARNING: non-finite gradients at step {step}; step skipped.")
                    if use_scaler:
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
                        "step": step, "loss": avg_loss, "lr": lr,
                        "perplexity": math.exp(min(avg_loss, 20)),
                        "tokens_per_second": tps, "mfu": mfu,
                        "tokens_seen": step * tokens_per_step,
                        "peak_memory_gb": peak_memory_gb(device),
                        "message": "Training.",
                    })
                    meter.reset()

                if eval_loader is not None and eval_every > 0 and step % eval_every == 0:
                    eval_loss = evaluate(active_model, eval_loader, device, autocast)
                    if eval_loss is None:
                        print("\n  eval skipped: validation split holds no complete window.")
                    else:
                        print(f"\n  eval loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
                        update_training_metrics(out, {"eval_loss": eval_loss, "message": "Evaluated."})
                        if eval_loss < best_eval:
                            best_eval = eval_loss
                            save_checkpoint(model, out, step, name=best_name, optimizer=optimizer, extra={"stage": "pretrained"})
                    meter.reset()

                if save_every > 0 and step % save_every == 0:
                    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"stage": "pretrained"})
                    meter.reset()

                if step >= total_steps:
                    done = True
                    break
        pbar.close()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving checkpoint before exit...")
        save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"stage": "pretrained"})
        update_training_metrics(out, {"status": "stopped", "message": "Interrupted by user."})
        print("Saved. Rerun the same command to resume from this step.")
        return
    finally:
        cleanup(device)
        pid_file.unlink(missing_ok=True)

    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"stage": "pretrained"})
    update_training_metrics(out, {"status": "complete", "message": "Pretraining complete."})
    print(f"Pretraining complete after {step:,} steps ({step * tokens_per_step:,} tokens). Saved to {out}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, autocast, max_batches: int = 50) -> Optional[float]:
    model.eval()
    total = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast:
            _, loss = model(input_ids, labels=labels)
        if loss is not None:
            total += float(loss.detach())
            count += 1
    model.train()
    return (total / count) if count else None
