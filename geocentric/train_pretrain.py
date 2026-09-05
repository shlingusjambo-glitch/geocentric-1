from __future__ import annotations

import json
import math
import os
import shlex
import sys
from contextlib import contextmanager
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
from geocentric.loss_guard import (
    OK,
    ROLLBACK,
    SKIP,
    STOP,
    LossGuard,
    LossGuardConfig,
    WeightSnapshot,
    choose_resume_checkpoint,
    record_checkpoint,
)
from geocentric.epicycle import (
    EpicycleConfig,
    EpicycleScheduler,
    build_epicycle_optimizer,
    equant_loss,
)
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
from geocentric.watermark import WatermarkConfig


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
    epicycle: EpicycleConfig | str | None = None,
    watermark: WatermarkConfig | None = None,
    loss_guard: bool = True,
    resume_from: str = "auto",
    snapshot_every: int = 200,
    stop_on_divergence: bool = False,
    loss_chunk_size: Optional[int] = None,
) -> None:
    if loss_chunk_size is not None and loss_chunk_size < 0:
        raise ValueError("loss_chunk_size must be nonnegative (0 disables chunking)")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    dtype = resolve_dtype(device, dtype_name)
    enable_fast_math()
    runtime_check(device, dtype)

    epi = epicycle if isinstance(epicycle, EpicycleConfig) else EpicycleConfig.preset(epicycle or "off")
    if loss_chunk_size is None:
        loss_chunk_size = 256 if epi.enabled and epi.armillary else 0

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
        watermark=watermark.to_dict() if watermark else None,
    )

    ckpt_name = pretrained_checkpoint_name(modelver)
    best_name = pretrained_checkpoint_name(modelver, best=True)
    start_step = 0
    resume_name, resume_why = (None, "")
    if resume:
        resume_name, resume_why = choose_resume_checkpoint(out, ckpt_name, best_name, resume_from)
    if resume_name and (out / resume_name).exists():
        print(f"Resuming from the {resume_why}.")
        model = load_checkpoint(out, device=device, dtype=torch.float32, checkpoint_name=resume_name)
        config = model.config
    else:
        print("Initializing model from random weights.")
        model = GeocentricGPT(config).to(device)
    if watermark is not None:
        # Kept on the live config as well as the loaded one, so resuming a run does
        # not silently drop the mark.
        model.config.watermark = watermark.to_dict()
        watermark.save(out)
        print(f"Watermark: on, identity {watermark.identity!r} "
              f"(gamma {watermark.gamma}, delta {watermark.delta})")

    # Master weights stay float32; autocast handles the low-precision compute. Casting
    # the parameters themselves to bf16 discards mantissa bits on every update and is
    # a large part of why small from-scratch models plateau early.
    model = model.to(device=device, dtype=torch.float32)
    model.loss_chunk_size = loss_chunk_size
    if gradient_checkpointing:
        # Users commonly add this flag after an OOM on an existing run. Loading
        # checkpoint config must not silently discard that new memory request.
        model.config.gradient_checkpointing = True
        for block in model.blocks:
            block.gradient_checkpointing = True
    if config.block_size != block_size:
        raise ValueError("Requested block_size differs from resumed checkpoint")

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
            optimizer_factory=(lambda: build_epicycle_optimizer(
                model, epi, learning_rate, weight_decay, quiet=True
            )) if epi.enabled and epi.armillary else None,
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
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    if not len(train_ds):
        raise ValueError("The training corpus has no complete window; add data or reduce block_size")
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    # drop_last stays off for eval: a small validation split can hold fewer windows
    # than one batch, and dropping it made evaluate() silently report a loss of 0.
    eval_kwargs = {**loader_kwargs, "drop_last": False}
    eval_loader = DataLoader(eval_ds, shuffle=False, **eval_kwargs) if eval_ds else None

    steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * max(1, epochs)
    warmup_steps = max(10, int(total_steps * warmup_ratio))

    if epi.enabled and epi.armillary:
        optimizer = build_epicycle_optimizer(model, epi, learning_rate, weight_decay)
    else:
        optimizer = build_optimizer(model, learning_rate, weight_decay, device_type=device.type)
    if resume_name and (out / resume_name).exists():
        start_step = load_optimizer_state(out, resume_name, optimizer)
        if start_step:
            print(f"Resuming at step {start_step:,} of {total_steps:,}.")

    active_model, compiled = maybe_compile(model, device, compile_mode)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if use_scaler and getattr(model, "_grad_scaler_state", None):
        scaler.load_state_dict(model._grad_scaler_state)
    model._grad_scaler = scaler
    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "mps"} and dtype != torch.float32
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    guard = LossGuard(
        LossGuardConfig(enabled=loss_guard, stop_on_divergence=stop_on_divergence),
        start_step=start_step,
    )
    # Snapshots make a rollback cost a couple of hundred steps instead of the whole
    # save interval. Host RAM, not VRAM.
    snapshot = WeightSnapshot(every=snapshot_every, enabled=loss_guard and snapshot_every > 0)
    if loss_guard:
        print(
            f"Loss guard: on | spikes beyond {guard.config.spike_sigma:.0f} sigma are dropped, "
            f"{guard.config.max_consecutive_spikes} in a row rolls back and re-warms over "
            f"{guard.config.rewarm_steps} steps"
        )

    epicycle_sched = EpicycleScheduler(epi, model, total_steps, block_size, start_step=start_step)
    if epi.enabled:
        gears = [n for n, on in (("DEFERENT", epi.deferent), ("HORIZON", epi.horizon),
                                 ("EQUANT", epi.equant), ("ARMILLARY", epi.armillary)) if on]
        print(f"EPICYCLE: {', '.join(gears) or 'no gears'} | {epicycle_sched.status(start_step)}")
        if compiled:
            print("  note: depth and context changes each trigger a torch.compile "
                  "recompilation. A handful over the run, not per step.")

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
            "epicycle": epi.to_dict(), "watermark_identity": watermark.identity if watermark else None,
            "loss_chunk_size": loss_chunk_size,
            "loss_guard": guard.config.to_dict(), "resume_from": resume_from,
            # Stored so tooling can show the exact line that resumes this run.
            "command": " ".join(shlex.quote(a) for a in [sys.executable, "-m", "geocentric.cli", *sys.argv[1:]]),
        },
    )

    # Publish the pid so monitoring tools identify this run unambiguously.
    pid_file = out / "trainer.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    restored_tokens = getattr(model, "_training_tokens", None)
    tokens_seen = restored_tokens if restored_tokens is not None else start_step * tokens_per_step
    if start_step and restored_tokens is None:
        print("Legacy checkpoint: historical token count is estimated from full-context steps.")
    update_training_metrics(out, {
        "step": start_step,
        "tokens_seen": tokens_seen,
        "message": f"Resuming at step {start_step:,}." if start_step else "Starting.",
    })

    meter = Throughput(n_params, block_size, device, dtype)
    step = start_step
    best_eval = float("inf")
    avg_loss = getattr(model, "_checkpoint_loss", None)
    running_loss = 0.0
    micro_count = 0
    window_tokens = 0
    model._training_tokens = tokens_seen
    _, ctx = epicycle_sched.apply(step)
    optimizer.zero_grad(set_to_none=True)

    try:
        pbar = tqdm(total=total_steps, initial=step, desc="pretrain", dynamic_ncols=True)
        done = step >= total_steps
        while not done:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                input_ids, labels = epicycle_sched.prepare_batch(input_ids, labels, ctx)

                selecting = epicycle_sched.selecting(step)
                with autocast:
                    _, loss = active_model(
                        input_ids, labels=labels,
                        loss_reduction="none" if selecting else "mean",
                        return_logits=False,
                    )
                if selecting:
                    # Report the honest mean, train on the trimmed hard band. Showing
                    # the selected loss instead would make the curve jump the moment
                    # selection turns on, for no reason the user could act on.
                    valid = labels.reshape(-1) != -100
                    reported = loss.detach()[valid].mean()
                    loss = equant_loss(loss, labels, epi.equant_keep, epi.equant_trim)
                else:
                    reported = loss.detach()
                if not torch.isfinite(loss):
                    print(f"WARNING: non-finite loss at step {step}; skipping microbatch.")
                    optimizer.zero_grad(set_to_none=True)
                    micro_count = 0
                    running_loss = 0.0
                    window_tokens = 0
                    continue

                # PackedDataset has no ignored labels before folding. Count on the
                # host to avoid an accelerator synchronization every microbatch.
                valid_tokens = (batch["labels"][:, :ctx].numel()
                                if epi.horizon_mode == "crop" else batch["labels"].numel())
                scaled = loss * (valid_tokens / tokens_per_step)
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                running_loss += reported * valid_tokens
                micro_count += 1
                window_tokens += valid_tokens
                meter.add(input_ids.numel())
                tokens_seen += valid_tokens
                model._training_tokens = tokens_seen

                if micro_count < gradient_accumulation_steps:
                    continue

                # Judged before the update lands, so a poisoned gradient can be dropped
                # rather than applied and then regretted.
                avg_loss = float(running_loss / max(1, window_tokens))
                verdict = guard.observe(step, avg_loss)

                if verdict.action in (SKIP, ROLLBACK, STOP):
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    micro_count = 0
                    window_tokens = 0

                if verdict.action == SKIP:
                    print(f"\n  [loss guard] step {step}: {verdict.reason}")
                    update_training_metrics(out, {
                        "message": f"Dropped a suspect update at step {step}.",
                        "loss_guard": guard.summary(),
                    })
                    continue

                if verdict.action == ROLLBACK:
                    restored = snapshot.restore(model)
                    if restored is None and (out / best_name).exists():
                        # Load into the existing module rather than building a new one:
                        # the optimizer holds references to these exact tensors, and
                        # rebinding `model` would leave it updating an orphan.
                        payload = torch.load(out / best_name, map_location="cpu",
                                             weights_only=False)
                        model.load_state_dict(
                            {k: v.to(device) for k, v in payload["model"].items()}
                        )
                        model._epicycle_state = payload.get("epicycle_state")
                        if "optimizer" in payload:
                            try:
                                optimizer.load_state_dict(payload["optimizer"])
                            except Exception as exc:
                                print(f"    optimizer state not restored ({exc}); "
                                      "its moments will decay out over the re-warm.")
                        restored = int(payload.get("step", 0))
                        del payload
                    print(f"\n  [loss guard] ROLLBACK at step {step}: {verdict.reason}")
                    if restored is None:
                        print("    nothing to roll back to yet; dropping the update and "
                              "re-warming the learning rate instead.")
                    else:
                        print(f"    restored the weights from step {restored:,}; "
                              f"re-warming the learning rate over "
                              f"{guard.config.rewarm_steps} steps.")
                        step = restored
                        saved_epi = getattr(model, "_epicycle_state", None)
                        epicycle_sched._grown_to = (saved_epi["grown_to"] if saved_epi
                                                   else epicycle_sched.active_layers(step))
                        # Weight-only snapshots must not keep potentially poisoned
                        # moments. Reset state while preserving the ring assignment.
                        if hasattr(optimizer, "_assign_rings"):
                            optimizer.state.clear()
                            optimizer._assign_rings()
                            optimizer._global_step = step
                        else:
                            optimizer.state.clear()
                        pbar.n = min(step, total_steps)
                        pbar.refresh()
                    update_training_metrics(out, {
                        "step": step, "message": f"Rolled back at step {step}.",
                        "loss_guard": guard.summary(),
                    })
                    _, ctx = epicycle_sched.apply(step)
                    continue

                if verdict.action == STOP:
                    print(f"\n  [loss guard] STOPPING: {verdict.reason}")
                    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                                    extra={"stage": "pretrained"})
                    update_training_metrics(out, {
                        "status": "diverged", "message": verdict.reason,
                        "loss_guard": guard.summary(),
                    })
                    return

                lr = lr_at_step(step, total_steps, learning_rate, warmup_steps, min_lr_ratio)
                lr *= verdict.lr_scale
                set_lr(optimizer, lr)
                if use_scaler:
                    scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(tokens_per_step / max(1, window_tokens))
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
                running_loss = 0.0
                micro_count = 0
                window_tokens = 0
                pbar.update(1)
                _, ctx = epicycle_sched.apply(step)
                snapshot.maybe_take(model, step, healthy=verdict.action == OK)

                if step % log_every == 0 or step == start_step + 1:
                    tps, mfu = meter.read()
                    status = " ".join(x for x in (epicycle_sched.status(step), guard.status()) if x)
                    line = format_progress(step, total_steps, avg_loss, lr, tps, mfu)
                    pbar.set_description(f"{line} | {status}" if status else line)
                    update_training_metrics(out, {
                        "step": step, "loss": avg_loss, "lr": lr,
                        "perplexity": math.exp(min(avg_loss, 20)),
                        "tokens_per_second": tps, "mfu": mfu,
                        "tokens_seen": tokens_seen,
                        "peak_memory_gb": peak_memory_gb(device),
                        "epicycle": epicycle_sched.status(step),
                        "loss_guard": guard.summary(),
                        "message": "Training.",
                    })
                    meter.reset()

                if eval_loader is not None and ((eval_every > 0 and step % eval_every == 0)
                                                or step >= total_steps):
                    # Evaluate at full depth and full context: an eval loss measured
                    # on a half-built model is not comparable across the run.
                    with _at_full_capacity(model, epicycle_sched):
                        eval_loss = evaluate(active_model, eval_loader, device, autocast)
                    if eval_loss is None:
                        print("\n  eval skipped: validation split holds no complete window.")
                    else:
                        print(f"\n  eval loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
                        update_training_metrics(out, {"step": step, "eval_loss": eval_loss, "message": "Evaluated."})
                        if eval_loss < best_eval:
                            best_eval = eval_loss
                            save_checkpoint(model, out, step, name=best_name, optimizer=optimizer,
                                            extra={"stage": "pretrained", "loss": avg_loss,
                                                   "eval_loss": eval_loss})
                            record_checkpoint(out, best_name, step, avg_loss, eval_loss)
                    meter.reset()

                if save_every > 0 and step % save_every == 0:
                    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                                    extra={"stage": "pretrained", "loss": avg_loss})
                    record_checkpoint(out, ckpt_name, step, avg_loss)
                    meter.reset()

                if step >= total_steps:
                    done = True
                    break
        pbar.close()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving checkpoint before exit...")
        save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                        extra={"stage": "pretrained", "loss": avg_loss})
        record_checkpoint(out, ckpt_name, step, avg_loss)
        update_training_metrics(out, {"status": "stopped", "message": "Interrupted by user.",
                                      "loss_guard": guard.summary()})
        print("Saved. Rerun the same command to resume from this step.")
        return
    finally:
        cleanup(device)
        pid_file.unlink(missing_ok=True)

    model.active_layers = None
    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                    extra={"stage": "pretrained", "loss": avg_loss})
    record_checkpoint(out, ckpt_name, step, avg_loss)
    update_training_metrics(out, {
        "status": "complete", "message": "Pretraining complete.",
        "step": step, "tokens_seen": tokens_seen,
        "loss": avg_loss,
        "epicycle_events": epicycle_sched.events,
        "loss_guard": guard.summary(),
    })
    if guard.spikes or guard.rollbacks:
        print(f"Loss guard: {len(guard.spikes)} spike(s), {guard.skipped_updates} update(s) "
              f"dropped, {len(guard.rollbacks)} rollback(s). Best loss {guard.best:.4f} "
              f"at step {guard.best_step:,}.")
    print(f"Pretraining complete after {step:,} steps ({tokens_seen:,} tokens). Saved to {out}")


@contextmanager
def _at_full_capacity(model, scheduler: EpicycleScheduler):
    """Temporarily undo DEFERENT so a measurement sees the whole model."""
    saved = model.active_layers
    model.active_layers = None
    try:
        yield
    finally:
        model.active_layers = saved


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
            _, loss = model(input_ids, labels=labels, return_logits=False, loss_reduction="sum")
        if loss is not None:
            total += float(loss.detach())
            count += int((labels != -100).sum())
    model.train()
    return (total / count) if count else None
