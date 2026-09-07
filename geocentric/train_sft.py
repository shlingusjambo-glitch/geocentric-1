from __future__ import annotations

import math
import json
import queue
import shutil
import shlex
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from geocentric.checkpoint import (
    find_tokenizer_path,
    load_checkpoint,
    load_optimizer_state,
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
from geocentric.loss_guard import (
    OK,
    ROLLBACK,
    SKIP,
    STOP,
    LossGuard,
    LossGuardConfig,
    WeightSnapshot,
    record_checkpoint,
)
from geocentric.training_metrics import initialize_training_metrics, update_training_metrics
from geocentric.watermark import WatermarkConfig
from tqdm.auto import tqdm


def _prefetch(iterable, buffer: int = 2):
    """Bounded, ordered CPU prefetch with cancellation and error propagation."""
    if buffer < 1:
        raise ValueError("prefetch buffer must be positive")
    q: queue.Queue = queue.Queue(maxsize=buffer)
    sentinel = object()
    errors: list[BaseException] = []
    cancelled = threading.Event()

    def put(item):
        while not cancelled.is_set():
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def worker() -> None:
        try:
            for item in iterable:
                if not put(item):
                    break
        except BaseException as exc:  # re-raised on the consuming thread
            errors.append(exc)
        finally:
            put(sentinel)

    thread = threading.Thread(target=worker, daemon=True, name="sft-prefetch")
    thread.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                if errors:
                    raise errors[0]
                return
            yield item
    finally:
        cancelled.set()
        # An external filesystem read may block independently; do not hang exit.
        thread.join(timeout=1.0)


def _normalize_gradients(parameters, factor):
    groups = {}
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad
            groups.setdefault((grad.device, grad.dtype), []).append(grad)
    for (device, _), gradients in groups.items():
        if device.type in ("cuda", "cpu") and all(not g.is_sparse for g in gradients):
            torch._foreach_mul_(gradients, factor)
        else:
            for grad in gradients:
                grad.mul_(factor)


def sft(
    model_dir: str,
    sft_data_path: str,
    output_dir: str | None = None,
    epochs: int = 3,
    batch_size: int = 1,
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
    compile_mode: str = "off",
    drop_overlong: bool = True,
    watermark: WatermarkConfig | None = None,
    drop_watermark: bool = False,
    loss_guard: bool = True,
    snapshot_every: int = 100,
    loss_chunk_size: int = 256,
    checkpoint_name: Optional[str] = None,
    optimizer_name: str = "auto",
    save_every: int = 100,
) -> None:
    if optimizer_name not in ("auto", "adamw", "capacity", "balanced"):
        raise ValueError("Unknown SFT optimizer")
    if save_every < 0:
        raise ValueError("save_every must be nonnegative")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if loss_chunk_size < 0 or gradient_accumulation_steps < 1:
        raise ValueError("loss_chunk_size must be nonnegative and accumulation positive")
    src = Path(model_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else src
    out.mkdir(parents=True, exist_ok=True)

    ckpt_name = sft_checkpoint_name(modelver)
    best_name = sft_checkpoint_name(modelver, best=True)
    resume_name = ckpt_name if not overwrite_output_dir and (out / ckpt_name).is_file() else None
    prior_metrics = {}
    metrics_path = out / "training_metrics.json"
    if metrics_path.is_file():
        try:
            prior_metrics = json.loads(metrics_path.read_text())
            if not isinstance(prior_metrics, dict):
                prior_metrics = {}
        except (OSError, json.JSONDecodeError):
            pass

    if not resume_name or not prior_metrics:
        initialize_training_metrics(out, phase="sft", config={"epochs": epochs})
    update_training_metrics(out, {"phase": "sft", "status": "preparing", "message": "Loading checkpoint on CPU."})
    device = select_device()
    dtype = resolve_dtype(device, dtype_name)
    enable_fast_math()
    runtime_check(device, dtype)

    if overwrite_output_dir:
        for old in out.glob("*_sft*.pt"):
            old.unlink(missing_ok=True)

    tokenizer_path = find_tokenizer_path(out if resume_name else src, extra_dirs=[src, out])
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer_path.resolve() != (out / "tokenizer.json").resolve():
        shutil.copy2(tokenizer_path, out / "tokenizer.json")
    pad_id = token_id(tokenizer, "<pad>")

    preferred = pretrained_checkpoint_name(modelver, best=True)
    load_dir = out if resume_name else src
    load_name = resume_name or checkpoint_name or (preferred if (src / preferred).is_file() else None)
    model = load_checkpoint(
        load_dir, device=torch.device("cpu"), dtype=torch.float32,
        checkpoint_name=load_name,
        mmap=True,
    )
    if resume_name:
        update_training_metrics(out, {"step": model._checkpoint_step,
                                      "message": f"Loaded SFT step {model._checkpoint_step:,}; checking token cache."})
    model.loss_chunk_size = loss_chunk_size
    model.loss_supervised_only = True
    if drop_watermark:
        model.config.watermark = None
        (out / "watermark.json").unlink(missing_ok=True)
        print("Watermark: removed.")
    elif watermark is not None:
        model.config.watermark = watermark.to_dict()
        watermark.save(out)
        print(f"Watermark: on, identity {watermark.identity!r}")
    elif model.config.watermark:
        print(f"Watermark: inherited from the pretrained checkpoint "
              f"({model.config.watermark.get('identity')!r}). Pass --no_watermark to drop it.")
    if device.type == "cuda":
        # Leave headroom for the display/compositor; no allocation-failure probing.
        torch.cuda.set_per_process_memory_fraction(0.75, torch.cuda.current_device())
        gradient_checkpointing = True
    model.config.gradient_checkpointing = gradient_checkpointing
    for block in model.blocks:
        block.gradient_checkpointing = gradient_checkpointing
    block_size = model.config.block_size

    update_training_metrics(out, {"message": "Checking the persistent SFT token cache."})
    dataset = SFTDataset(tokenizer, sft_data_path, block_size=block_size, drop_overlong=drop_overlong,
                         cache_dir=out / "sft-cache")
    resume_state = getattr(model, "_sft_resume_state", None) if resume_name else None
    geometry = {"cache_key": dataset.examples.path.stem, "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps, "eval_ratio": eval_ratio,
                "learning_rate": learning_rate, "warmup_ratio": warmup_ratio,
                "min_lr_ratio": min_lr_ratio}
    if resume_state and resume_state.get("geometry") != geometry:
        raise ValueError("SFT resume data or training settings changed. Use the original settings "
                         "to resume, or a new output directory for a new fine-tune.")
    eval_len = max(1, int(len(dataset) * eval_ratio)) if len(dataset) > 50 and eval_ratio > 0 else 0
    if eval_len:
        train_ds, eval_ds = random_split(
            dataset, [len(dataset) - eval_len, eval_len],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds, eval_ds = dataset, None

    if num_workers is None:
        num_workers = 0
    if num_workers != 0:
        raise ValueError("Disk-backed SFT currently requires --num_workers 0")
    update_training_metrics(out, {"message": "Moving model to training device."})
    from geocentric.epicycle import EpicycleConfig, build_epicycle_optimizer
    prior_optimizer = getattr(model, "_sft_optimizer_config", None)
    legacy_adamw_resume = (resume_name and prior_optimizer is None
                           and getattr(model, "_checkpoint_optimizer_type", None) == "AdamW")
    inherited = EpicycleConfig.from_dict(
        {} if legacy_adamw_resume else prior_optimizer if prior_optimizer is not None else
        (getattr(model, "_epicycle_state", None) or {}).get("config"))
    optimizer_config = (inherited if optimizer_name == "auto" and inherited.enabled and inherited.armillary
                        else EpicycleConfig.preset(optimizer_name) if optimizer_name in ("capacity", "balanced") else None)
    static_bytes = sum(p.numel() for p in model.parameters()) * 16
    if optimizer_config is not None:
        probe = build_epicycle_optimizer(model, optimizer_config, learning_rate, weight_decay, quiet=True)
        second = sum(4 * (p.numel() // p.shape[-1] + p.shape[-1])
                     if optimizer_config.armillary_factored and p.ndim >= 2 and min(p.shape) > 1
                     else 2 * p.numel() for p in model.parameters())
        static_bytes = sum(p.numel() for p in model.parameters()) * 8 + second + 4 * max(probe.ring_load)
        del probe
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        # FP32 weights + gradients + Adam moments, before any activations/workspace.
        budget = min(total * .75, max(0, free - 512 * 1024**2))
        if static_bytes >= budget:
            update_training_metrics(out, {"status": "failed", "message": "Insufficient VRAM for selected SFT optimizer."})
            raise RuntimeError("SFT weights/gradients/moments exceed the available VRAM budget. "
                               "Pretraining with compact EPICYCLE state can fit while full SFT cannot. "
                               "Try --optimizer capacity, or a smaller model; batch size cannot remove optimizer state.")
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        update_training_metrics(out, {"status": "failed", "message": "Insufficient device memory to load SFT model."})
        raise
    print(f"SFT memory settings: batch={batch_size}, loss chunks={loss_chunk_size}, "
          f"checkpointing={gradient_checkpointing}, workers=0, compile={compile_mode}", flush=True)
    collate = PadCollate(pad_id)
    loader_kwargs = dict(
        batch_size=batch_size, collate_fn=collate, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs) if eval_ds else None

    batches_per_epoch = math.ceil(len(train_ds) / batch_size)
    if not batches_per_epoch:
        raise ValueError("No training conversations remain after filtering")
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
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

    optimizer = (build_epicycle_optimizer(model, optimizer_config, learning_rate, weight_decay)
                 if optimizer_config is not None else
                 build_optimizer(model, learning_rate, weight_decay, device_type=device.type))
    start_step = load_optimizer_state(out, resume_name, optimizer, mmap=True) if resume_name else 0
    if resume_name and not resume_state:
        print("Legacy SFT checkpoint: restoring weights/optimizer/step; old shuffled batch order "
              "was not saved, so the remaining data position is approximate.", flush=True)
    if start_step:
        print(f"Resuming SFT at step {start_step:,} of {total_steps:,}.", flush=True)
    active_model, compiled = maybe_compile(model, device, compile_mode)
    print("SFT assistant-only vocabulary projection: " +
          ("on" if loss_chunk_size > 0 and not compiled else "inactive (requires chunked loss and compile off)"))
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if resume_name and use_scaler and getattr(model, "_grad_scaler_state", None):
        scaler.load_state_dict(model._grad_scaler_state)
    model._grad_scaler = scaler
    loss_normalizer = batch_size * block_size * gradient_accumulation_steps
    autocast = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "mps"} and dtype != torch.float32
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    metrics_config = {
            "examples": len(dataset), "dropped_overlong": getattr(dataset, "dropped", 0),
            "epochs": epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "total_steps": total_steps,
            "block_size": block_size, "dtype": str(dtype), "compiled": compiled,
            "loss_chunk_size": loss_chunk_size, "optimizer": type(optimizer).__name__,
            "supervised_only_projection": loss_chunk_size > 0 and not compiled,
            "optimizer_config": optimizer_config.to_dict() if optimizer_config else None,
            "watermark_identity": (model.config.watermark or {}).get("identity"),
            "loss_guard": LossGuardConfig(enabled=loss_guard).to_dict(),
            "params": n_params, "n_layer": model.config.n_layer,
            "n_embd": model.config.n_embd,
            "command": shlex.join([sys.executable, "-m", "geocentric.cli", *sys.argv[1:]]),
            # Nominal, not exact: conversations vary in length, so actual supervised
            # tokens per step (tracked below as tokens_seen) drift from this budget.
            "tokens_per_step": loss_normalizer,
        }
    saved_tokens = (resume_state.get("tokens_seen", 0) if resume_state else
                    prior_metrics.get("tokens_seen", 0)) if resume_name else 0
    if resume_name:
        update_training_metrics(out, {"status": "running", "step": start_step,
                                      "tokens_seen": int(saved_tokens or 0),
                                      "config": metrics_config,
                                      "message": f"Resuming SFT at step {start_step:,}."})
    else:
        initialize_training_metrics(out, phase="sft", config=metrics_config)

    # Fine-tuning is short and its loss is noisier per step than pretraining's, so the
    # guard earns its place here mostly by dropping the occasional poisoned batch
    # rather than by rolling back.
    guard = LossGuard(LossGuardConfig(enabled=loss_guard, warmup_steps=30), start_step=start_step)
    if resume_state and resume_state.get("guard") and loss_guard:
        guard.__dict__.update(resume_state["guard"])
    snapshot = WeightSnapshot(every=snapshot_every, enabled=loss_guard and snapshot_every > 0)
    meter = Throughput(n_params, block_size, device, dtype)
    step = start_step
    tokens_seen = int(saved_tokens or 0)
    model._training_tokens = tokens_seen
    best_eval = float((resume_state or {}).get("best_eval", prior_metrics.get("best_eval_loss") or "inf"))
    running_loss = 0.0
    micro_count = 0
    window_tokens = 0
    avg_loss = None
    optimizer.zero_grad(set_to_none=True)

    def remember_position(epoch, next_batch):
        model._sft_resume_state = {
            "epoch": epoch, "next_batch": next_batch, "geometry": geometry,
            "tokens_seen": tokens_seen, "best_eval": best_eval,
            "device_type": device.type,
            "guard": {k: v for k, v in vars(guard).items() if k != "config"},
            "cpu_rng": torch.get_rng_state(),
            "device_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else
                           torch.mps.get_rng_state() if device.type == "mps" else None),
        }

    if resume_state:
        torch.set_rng_state(resume_state["cpu_rng"])
        if resume_state.get("device_rng") is not None and resume_state.get("device_type") == device.type:
            if device.type == "cuda":
                torch.cuda.set_rng_state(resume_state["device_rng"], device)
            elif device.type == "mps":
                torch.mps.set_rng_state(resume_state["device_rng"])
    first_epoch = resume_state["epoch"] if resume_state else step // steps_per_epoch + 1
    first_batch = resume_state["next_batch"] if resume_state else (step % steps_per_epoch) * gradient_accumulation_steps
    remember_position(first_epoch, first_batch)

    batches = None
    pbar = None
    try:
        pbar = tqdm(total=total_steps, initial=min(step, total_steps), desc="sft", dynamic_ncols=True)
        for epoch in range(first_epoch, max(1, epochs) + 1):
            epoch_start_step = step
            generator = torch.Generator().manual_seed(42 + epoch)
            train_loader = DataLoader(train_ds, shuffle=True, drop_last=False,
                                      generator=generator, **loader_kwargs)
            skip_batches = first_batch if epoch == first_epoch else 0
            model.train()
            batches = _prefetch(train_loader)
            for batch_index, batch in enumerate(batches):
                if batch_index < skip_batches:
                    continue
                # Labels are already on CPU: counting here avoids a GPU reduction
                # and host synchronization after each backward microbatch.
                supervised_indices = (batch["labels"].reshape(-1) != -100).nonzero(as_tuple=True)[0]
                supervised_tokens = supervised_indices.numel()
                supervised_indices = supervised_indices.to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast:
                    _, loss = active_model(input_ids, labels=labels, return_logits=False,
                                           loss_reduction="sum", supervised_indices=supervised_indices)
                scaled = loss / loss_normalizer
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                # Enqueue backward before the loss readback, avoiding a host
                # barrier between forward and backward on every microbatch.
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    optimizer.zero_grad(set_to_none=True)
                    micro_count = 0
                    running_loss = 0.0
                    window_tokens = 0
                    continue

                running_loss += loss_value
                micro_count += 1
                window_tokens += supervised_tokens
                meter.add(input_ids.numel())
                if micro_count < gradient_accumulation_steps and batch_index + 1 < len(train_loader):
                    continue

                if window_tokens == 0:
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    micro_count = 0
                    continue
                avg_loss = running_loss / window_tokens
                verdict = guard.observe(step, avg_loss)
                if verdict.action in (SKIP, ROLLBACK, STOP):
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    micro_count = 0
                    window_tokens = 0
                    if verdict.action == SKIP:
                        print(f"\n  [loss guard] step {step}: {verdict.reason}")
                        continue
                    if verdict.action == STOP:
                        save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                                        extra={"sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}, "stage": "sft", "loss": avg_loss})
                        update_training_metrics(out, {"status": "diverged", "message": verdict.reason})
                        return
                    restored = snapshot.restore(model)
                    print(f"\n  [loss guard] {verdict.action.upper()} at step {step}: "
                          f"{verdict.reason}")
                    if restored is not None:
                        print(f"    restored the weights from step {restored:,}.")
                        optimizer.state.clear()
                    continue

                lr = lr_at_step(step, total_steps, learning_rate, warmup_steps, min_lr_ratio)
                lr *= verdict.lr_scale
                set_lr(optimizer, lr)
                if use_scaler:
                    scaler.unscale_(optimizer)
                # Losses were summed, then divided by a nominal token budget
                # before backward to keep FP16 gradients in range. Normalize by actual supervised tokens, including
                # the final partial window and variable-length assistant responses.
                _normalize_gradients(model.parameters(), loss_normalizer / window_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                healthy_gradients = bool(torch.isfinite(grad_norm))
                if healthy_gradients:
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                elif use_scaler:
                    scaler.update()

                if not healthy_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    micro_count = 0
                    window_tokens = 0
                    # No optimizer update occurred: do not advance its schedule
                    # or count the rejected tokens as trained.
                    remember_position(epoch, batch_index + 1)
                    continue

                optimizer.zero_grad(set_to_none=True)
                step += 1
                tokens_seen += window_tokens
                model._training_tokens = tokens_seen
                running_loss = 0.0
                micro_count = 0
                window_tokens = 0
                pbar.update(1)
                snapshot.maybe_take(model, step, healthy=verdict.action == OK)
                remember_position(epoch, batch_index + 1)

                if step % log_every == 0:
                    tps, mfu = meter.read()
                    pbar.set_description(format_progress(step, total_steps, avg_loss, lr, tps, mfu))
                    update_training_metrics(out, {
                        "step": step, "epoch": epoch, "loss": avg_loss, "lr": lr,
                        "perplexity": math.exp(min(avg_loss, 20)),
                        "tokens_per_second": tps, "loss_guard": guard.summary(),
                        "tokens_seen": tokens_seen,
                        "message": "Training.",
                    })
                    meter.reset()

                if save_every and step % save_every == 0 and step < total_steps:
                    save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                                    extra={"stage": "sft", "loss": avg_loss,
                                           "sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}})
                    record_checkpoint(out, ckpt_name, step, avg_loss)

            if step == epoch_start_step and skip_batches < len(train_loader):
                raise RuntimeError("SFT epoch made no optimizer updates; check non-finite losses, gradients and supervision")
            if eval_loader is not None:
                eval_loss = evaluate(active_model, eval_loader, device, autocast)
                print(f"\n  epoch {epoch} eval loss {eval_loss:.4f} | ppl {math.exp(min(eval_loss, 20)):.2f}")
                update_training_metrics(out, {"eval_loss": eval_loss, "message": f"Epoch {epoch} evaluated."})
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    remember_position(epoch + 1, 0)
                    save_checkpoint(model, out, step, name=best_name, optimizer=optimizer,
                                    extra={"sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}, "stage": "sft", "loss": avg_loss, "eval_loss": eval_loss})
                    record_checkpoint(out, best_name, step, avg_loss, eval_loss)
                    print("  new best SFT checkpoint saved.")
                meter.reset()
            remember_position(epoch + 1, 0)
            save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer,
                            extra={"sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}, "stage": "sft", "loss": avg_loss})
            record_checkpoint(out, ckpt_name, step, avg_loss)
        pbar.close()
    except torch.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        update_training_metrics(out, {"status": "failed", "message": "SFT ran out of memory; source checkpoint retained."})
        raise RuntimeError("SFT ran out of memory. Source checkpoint is unchanged. "
                           "Use batch_size=1, loss_chunk_size=64, compile off; close the inference server.") from None
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving SFT checkpoint before exit...")
        save_checkpoint(model, out, step, name=ckpt_name, optimizer=optimizer, extra={"sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}, "stage": "sft"})
        update_training_metrics(out, {"status": "stopped", "step": step, "tokens_seen": tokens_seen,
                                      "message": "Interrupted by user."})
        return
    except Exception as exc:
        try:
            update_training_metrics(out, {"status": "failed", "step": step,
                                          "message": f"SFT failed: {type(exc).__name__}: {exc}"})
        except Exception as metrics_error:
            print(f"Could not record failure status: {metrics_error}", file=sys.stderr)
        raise
    finally:
        if batches is not None:
            batches.close()
        if pbar is not None:
            pbar.close()
        cleanup(device)

    if not (out / best_name).exists():
        save_checkpoint(model, out, step, name=best_name, optimizer=optimizer, extra={"sft_optimizer_config": optimizer_config.to_dict() if optimizer_config else {}, "stage": "sft"})
    update_training_metrics(out, {"status": "complete", "step": step, "tokens_seen": tokens_seen,
                                  "message": "SFT complete.",
                                  "loss_guard": guard.summary()})
    print(f"SFT complete after {step:,} steps. Saved to {out}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, autocast) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        for batch in loader:
            selected = (batch["labels"].reshape(-1) != -100).nonzero(as_tuple=True)[0]
            supervised_tokens = selected.numel()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast:
                _, loss = model(input_ids, labels=labels, return_logits=False, loss_reduction="sum",
                                supervised_indices=selected.to(device, non_blocking=True))
            value = float(loss.detach()) if loss is not None else float('nan')
            if not math.isfinite(value):
                raise RuntimeError("Non-finite SFT validation loss; refusing to select a best checkpoint")
            total += value
            count += supervised_tokens
        if count == 0:
            raise RuntimeError("SFT validation contains no supervised tokens")
        return total / count
    finally:
        model.train(was_training)
