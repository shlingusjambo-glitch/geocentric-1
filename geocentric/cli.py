from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geocentric import __version__

BANNER = r"""
   ____                            _        _
  / ___| ___  ___   ___ ___ _ __ | |_ _ __(_) ___
 | |  _ / _ \/ _ \ / __/ _ \ '_ \| __| '__| |/ __|
 | |_| |  __/ (_) | (_|  __/ | | | |_| |  | | (__
  \____|\___|\___/ \___\___|_| |_|\__|_|  |_|\___|
"""


def _add_watermark_flags(p: argparse.ArgumentParser) -> None:
    """Flags that decide whether the command asks about watermarking at all.

    With none of them set, an interactive terminal gets asked and a pipe does not —
    a training run started from a script must never block on a prompt nobody is
    there to answer.
    """
    group = p.add_argument_group("watermarking")
    group.add_argument("--watermark", action="store_true",
                       help="Watermark output, identified by --modelver. Skips the prompt.")
    group.add_argument("--watermark_identity", default=None, metavar="NAME",
                       help="Watermark output and identify it as NAME. Skips the prompt.")
    group.add_argument("--no_watermark", action="store_true",
                       help="Do not watermark, and drop any inherited watermark. Skips the prompt.")
    group.add_argument("--watermark_gamma", type=float, default=None,
                       help="Green-list fraction (default 0.25).")
    group.add_argument("--watermark_delta", type=float, default=None,
                       help="Logit bias on green tokens (default 2.0). Higher is more "
                            "detectable and costs more output quality.")
    group.add_argument("--yes", "-y", action="store_true",
                       help="Answer every interactive prompt with its default.")


def _add_loss_guard_flags(p: argparse.ArgumentParser, resume: bool = True) -> None:
    group = p.add_argument_group("loss stability")
    group.add_argument("--no_loss_guard", action="store_true",
                       help="Disable spike detection and rollback. The loss then rises "
                            "wherever it rises and nothing intervenes.")
    group.add_argument("--snapshot_every", type=int, default=200 if resume else 100,
                       help="Steps between known-good weight snapshots kept in host RAM "
                            "(4 bytes/param). 0 rolls back to the last saved checkpoint "
                            "instead, which is free but loses more progress.")
    if resume:
        group.add_argument("--resume_from", default="auto", choices=["auto", "last", "best"],
                           help="Which checkpoint a resumed run continues from. auto takes "
                                "the best one only when the recorded loss says it is better.")
        group.add_argument("--stop_on_divergence", action="store_true",
                           help="Exit instead of rolling back when the loss has stayed above "
                                "its best for hundreds of steps.")


def _add_common_training_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~30%% speed for a large memory saving. Use when you hit OOM.")
    p.add_argument("--compile", dest="compile_mode", default="auto", choices=["auto", "off"])
    p.add_argument("--loss_chunk_size", type=int, default=None,
                   help="Tokens per checkpointed vocabulary projection; 0 uses dense loss. "
                        "Default: 256 for EPICYCLE memory/full/capacity; otherwise dense.")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--modelver", default="Geocentric")
    p.add_argument("--overwrite_output_dir", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geocentric",
        description="Geocentric — train a causal language model from scratch.",
    )
    parser.add_argument("--version", action="version", version=f"Geocentric {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    tok = sub.add_parser("train-tokenizer", help="Train tokenizer.json from a local corpus")
    tok.add_argument("--data_path", required=True)
    tok.add_argument("--output", default="runs/geocentric/tokenizer.json")
    tok.add_argument("--vocab_size", type=int, default=32000)
    tok.add_argument("--doc_sep", default=None)

    prep = sub.add_parser("prepare", help="Tokenize a corpus into binary token shards")
    prep.add_argument("--data_path", required=True)
    prep.add_argument("--output_dir", default="runs/geocentric")
    prep.add_argument("--tokenizer", default=None)
    prep.add_argument("--val_fraction", type=float, default=0.005)
    prep.add_argument("--doc_sep", default=None)

    pre = sub.add_parser("pretrain", help="Pretrain a model from random initialization")
    pre.add_argument("--data_path", required=True)
    pre.add_argument("--output_dir", default="runs/geocentric")
    pre.add_argument("--preset", default="120m", help="Parameter budget, e.g. 50m, 120m, 350m, 1b")
    pre.add_argument("--vocab_size", type=int, default=32000)
    pre.add_argument("--block_size", type=int, default=None, help="Context length (default from preset)")
    pre.add_argument("--n_layer", type=int, default=None)
    pre.add_argument("--n_head", type=int, default=None)
    pre.add_argument("--n_kv_head", type=int, default=None)
    pre.add_argument("--n_embd", type=int, default=None)
    pre.add_argument("--dropout", type=float, default=0.0)
    pre.add_argument("--epochs", type=int, default=1)
    pre.add_argument("--max_steps", type=int, default=0)
    pre.add_argument("--warmup_ratio", type=float, default=0.01)
    pre.add_argument("--doc_sep", default=None,
                     help="Document separator in plain text. Omit to treat each file as one document.")
    pre.add_argument("--tokenizer", dest="tokenizer_path", default=None)
    pre.add_argument("--eval_every", type=int, default=500)
    pre.add_argument("--save_every", type=int, default=1000)
    pre.add_argument("--no_resume", action="store_true")
    pre.add_argument("--reprepare", action="store_true", help="Re-tokenize the corpus even if shards exist")
    pre.add_argument("--epicycle", default="off",
                     choices=["off", "speed", "quality", "memory", "full", "capacity", "balanced", "selective"],
                     help="EPICYCLE training gears. speed = elastic depth + context; "
                          "quality = adds token selection; memory = adds rotating optimizer "
                          "state so more parameters fit; full = everything.")
    _add_common_training_flags(pre)
    _add_loss_guard_flags(pre)
    _add_watermark_flags(pre)

    ft = sub.add_parser("sft", help="Instruction fine-tune a pretrained checkpoint")
    ft.add_argument("--model_dir", default="runs/geocentric")
    ft.add_argument("--sft_data_path", required=True)
    ft.add_argument("--output_dir", default=None)
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--warmup_ratio", type=float, default=0.03)
    ft.add_argument("--keep_overlong", action="store_true",
                    help="Truncate conversations longer than the context instead of dropping them")
    _add_common_training_flags(ft)
    _add_loss_guard_flags(ft, resume=False)
    _add_watermark_flags(ft)

    pl = sub.add_parser("pipeline", help="Pretrain then SFT in one command")
    pl.add_argument("--data_path", required=True)
    pl.add_argument("--sft_data_path", required=True)
    pl.add_argument("--output_dir", default="runs/geocentric")
    pl.add_argument("--preset", default="120m")
    pl.add_argument("--vocab_size", type=int, default=32000)
    pl.add_argument("--block_size", type=int, default=None)
    pl.add_argument("--epochs", type=int, default=1)
    pl.add_argument("--sft_epochs", type=int, default=3)
    pl.add_argument("--sft_learning_rate", type=float, default=None,
                    help="Fine-tuning rate (default 1e-4). Independent of --learning_rate.")
    pl.add_argument("--max_steps", type=int, default=0)
    pl.add_argument("--doc_sep", default=None)
    pl.add_argument("--epicycle", default="off",
                    choices=["off", "speed", "quality", "memory", "full", "capacity", "balanced", "selective"])
    _add_common_training_flags(pl)
    _add_loss_guard_flags(pl)
    _add_watermark_flags(pl)

    vis = sub.add_parser("train-vision", help="Make a text model multimodal on image/text pairs")
    vis.add_argument("--model_dir", default="runs/geocentric")
    vis.add_argument("--vision_data_path", required=True,
                     help=".json/.jsonl of {image, caption} or {image, messages} records")
    vis.add_argument("--output_dir", default=None)
    vis.add_argument("--image_root", default=None, help="Directory image paths are relative to")
    vis.add_argument("--epochs", type=int, default=3)
    vis.add_argument("--learning_rate", type=float, default=2e-4)
    vis.add_argument("--projector_lr_multiplier", type=float, default=5.0,
                     help="The projector is the only randomly-initialized bridge between two "
                          "spaces; it needs a higher rate than the decoder.")
    vis.add_argument("--image_size", type=int, default=224)
    vis.add_argument("--patch_size", type=int, default=16)
    vis.add_argument("--vision_layers", type=int, default=6)
    vis.add_argument("--vision_width", type=int, default=384)
    vis.add_argument("--vision_pool", type=int, default=2,
                     help="Average-pool the patch grid NxN before projecting. 2 turns a 14x14 "
                          "grid into 49 image tokens instead of 196.")
    vis.add_argument("--freeze_lm", action="store_true",
                     help="Train only the vision tower and projector, leaving the language "
                          "model untouched. The usual stage-1 pass.")
    vis.add_argument("--batch_size", type=int, default=4)
    vis.add_argument("--gradient_accumulation_steps", type=int, default=8)
    vis.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    vis.add_argument("--loss_chunk_size", type=int, default=0)
    vis.add_argument("--num_workers", type=int, default=None)
    vis.add_argument("--modelver", default="Geocentric")
    _add_watermark_flags(vis)

    bench = sub.add_parser("bench", aliases=["parallax"],
                           help="Run the PARALLAX benchmark suite and write a Markdown report")
    bench.add_argument("--model_dir", default="runs/geocentric")
    bench.add_argument("--output", default=None,
                       help="Where to write the report (default <model_dir>/PARALLAX.md)")
    bench.add_argument("--eval_text", default=None,
                       help="Path to your own held-out text. Strongly preferred over the "
                            "bundled probes, which are only a few kilobytes.")
    bench.add_argument("--vision_data", default=None,
                       help="Held-out image/text pairs, for the PRISM grounding probe")
    bench.add_argument("--only", nargs="+", default=None, metavar="PROBE",
                       help="Run only these probes: ZENITH MERIDIAN SEXTANT ASTROLABE NADIR "
                            "ORBIT PRISM")
    bench.add_argument("--checkpoint", default=None)
    bench.add_argument("--nadir_tokens", type=int, default=128)
    bench.add_argument("--quiet", action="store_true", help="Write the report, print nothing")

    rel = sub.add_parser("release", help="Package a checkpoint for public release")
    rel.add_argument("--model_dir", default="runs/geocentric")
    rel.add_argument("--output_dir", required=True)
    rel.add_argument("--checkpoint", default=None)
    rel.add_argument("--license", dest="license_name", default="",
                     help="Licence to state in the model card, e.g. apache-2.0")
    rel.add_argument("--description", default="", help="One-line description for the model card")
    rel.add_argument("--no_benchmark", action="store_true",
                     help="Skip PARALLAX. The model card then makes no quality claim.")
    rel.add_argument("--eval_text", default=None)
    rel.add_argument("--vision_data", default=None)
    rel.add_argument("--overwrite", action="store_true")
    _add_watermark_flags(rel)

    det = sub.add_parser("detect", help="Test whether text carries a Geocentric watermark")
    det.add_argument("--text", default=None, help="Text to test. Omit to read stdin.")
    det.add_argument("--file", default=None, help="File to test")
    det.add_argument("--identity", nargs="+", default=None, metavar="NAME",
                     help="Identities to test against. Defaults to the model_dir's own.")
    det.add_argument("--model_dir", default="runs/geocentric",
                     help="Supplies the tokenizer, and the default identity")
    det.add_argument("--z_threshold", type=float, default=4.0)
    det.add_argument("--no_capacity", action="store_true",
                     help="Skip the capacity check. That check loads the model to explain a "
                          "negative result, and is the difference between 'no watermark' and "
                          "'this text was too predictable to carry one'.")
    det.add_argument("--gamma", type=float, default=0.25)
    det.add_argument("--delta", type=float, default=2.0)

    ch = sub.add_parser("chat", aliases=["try"], help="Test a checkpoint interactively")
    ch.add_argument("--model_dir", default="runs/geocentric")
    ch.add_argument("--terminal", action="store_true", help="Use the terminal instead of the web interface for try")
    ch.add_argument("--web", action="store_true", help="Launch the web interface with chat too")
    ch.add_argument("--host", default="0.0.0.0", help="Server bind address (default: local network)")
    ch.add_argument("--port", type=int, default=8000)
    ch.add_argument("--no_browser", action="store_true")
    ch.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ch.add_argument("--checkpoint", default=None)
    ch.add_argument("--max_new_tokens", type=int, default=256)
    ch.add_argument("--temperature", type=float, default=0.8)
    ch.add_argument("--top_k", type=int, default=50)
    ch.add_argument("--top_p", type=float, default=0.95)
    ch.add_argument("--min_p", type=float, default=0.05)
    ch.add_argument("--repetition_penalty", type=float, default=1.25,
                    help="Below ~1.2 a partly-trained model collapses into loops: measured "
                         "at step 5,231, penalty 1.1 gave a 0.51 unique-token ratio and "
                         "7 repeats of the same bigram; 1.25 gave 0.85 and 2.")
    ch.add_argument("--system", default=None)
    ch.add_argument("--image", default=None,
                    help="Image to attach to the first turn (multimodal checkpoints only)")
    ch.add_argument("--mode", default="auto", choices=["auto", "chat", "base"],
                    help="auto picks chat for an instruction-tuned checkpoint and raw "
                         "continuation for a pretrained-only one")

    gen = sub.add_parser("generate", help="Generate a single completion and exit")
    gen.add_argument("--model_dir", default="runs/geocentric")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max_new_tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--raw", action="store_true", help="Force raw completion, no chat template")
    gen.add_argument("--image", default=None, help="Image to condition on (multimodal only)")

    pc = sub.add_parser("plan", help="Show the architecture and token budget for a parameter target")
    pc.add_argument("--preset", default="120m")
    pc.add_argument("--vocab_size", type=int, default=32000)
    pc.add_argument("--block_size", type=int, default=None)

    sub.add_parser("download-wiki", help="Download WikiText-103 pretraining data")
    sub.add_parser("download-alpaca", help="Download a cleaned Alpaca instruction set")
    sub.add_parser("list-models", help="List local checkpoints under runs/ and models/")

    align = sub.add_parser("align-safety", help="Optional post-SFT harm-focused refusal and uncertainty training")
    align.add_argument("--model_dir", required=True)
    align.add_argument("--output_dir", required=True)
    align.add_argument("--data_path", required=True, help="Reviewed JSONL with refusal, benign, and uncertainty categories")
    align.add_argument("--checkpoint", default=None)
    align.add_argument("--epochs", type=int, default=1)
    align.add_argument("--batch_size", type=int, default=1)
    align.add_argument("--gradient_accumulation_steps", type=int, default=4)
    align.add_argument("--learning_rate", type=float, default=1e-5)
    align.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    align.add_argument("--yes", action="store_true", help="Explicitly accept the printed training plan")

    behavior = sub.add_parser("check-behavior", help="Generate auditable held-out factuality/behavior comparisons")
    behavior.add_argument("--model_dir", required=True)
    behavior.add_argument("--compare_dir", default=None)
    behavior.add_argument("--data_path", required=True)
    behavior.add_argument("--output", required=True)
    behavior.add_argument("--max_new_tokens", type=int, default=128)

    for training_parser in (pre, pl):
        training_parser.add_argument("--equant_sparse_replay", action="store_true",
                                     help="Replay only selected vocabulary rows; requires an EQUANT preset and eager mode")
    return parser


def _dims_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from geocentric.param_compiler import compute_fluid_dimensions

    dims = compute_fluid_dimensions(args.preset, vocab_size=args.vocab_size,
                                    block_size=getattr(args, "block_size", None))
    overridden = False
    for key in ("n_layer", "n_head", "n_kv_head", "n_embd", "block_size"):
        override = getattr(args, key, None)
        if override:
            dims[key] = override
            overridden = True
    if overridden:
        # Recompute rather than reporting the preset's count for a hand-edited shape.
        from geocentric.param_compiler import exact_params

        dims["params"] = exact_params(
            args.vocab_size, dims["n_layer"], dims["n_embd"], dims["n_head"], dims["n_kv_head"]
        )
    return dims


def _auto_batch(args: argparse.Namespace, dims: dict[str, Any]) -> tuple[int, int]:
    """Return explicit sizes, or zeros meaning 'measure it on the real model'.

    A formula cannot predict activation memory reliably enough to trust a multi-day
    run to it, so unless the user pins a size the trainer probes the actual model
    on the actual device.
    """
    if args.batch_size and args.gradient_accumulation_steps:
        return args.batch_size, args.gradient_accumulation_steps
    if args.batch_size:
        return args.batch_size, args.gradient_accumulation_steps or 0
    return 0, args.gradient_accumulation_steps or 0


def _run_pretrain(args: argparse.Namespace, watermark=None) -> None:
    from geocentric.train_pretrain import pretrain
    from geocentric.watermark import resolve_watermark

    if watermark is None and not getattr(args, "_watermark_resolved", False):
        watermark, _ = resolve_watermark(
            args, args.modelver,
            context="this model is being trained from scratch, so now is the moment to "
                    "decide how its output should be attributed",
        )

    dims = _dims_from_args(args)
    batch, accum = _auto_batch(args, dims)
    print(f"Architecture: {dims['n_layer']}L x {dims['n_embd']}d x {dims['n_head']}h "
          f"(kv {dims['n_kv_head']}) ctx {dims['block_size']} ≈ {dims['params']:,} params")
    if batch == 0:
        print("Batch size: measuring on device...")

    from geocentric.epicycle import EpicycleConfig
    epi = EpicycleConfig.preset(getattr(args, "epicycle", "off"))
    if getattr(args, "equant_sparse_replay", False):
        if not epi.enabled or not epi.equant:
            raise ValueError("--equant_sparse_replay requires quality, selective, or full")
        epi.equant_sparse_replay = True

    pretrain(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        block_size=dims["block_size"],
        n_layer=dims["n_layer"],
        n_head=dims["n_head"],
        n_kv_head=dims["n_kv_head"],
        n_embd=dims["n_embd"],
        dropout=args.dropout,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.learning_rate or 6e-4,
        warmup_ratio=args.warmup_ratio,
        doc_sep=args.doc_sep,
        dtype_name=args.dtype,
        tokenizer_path=args.tokenizer_path,
        gradient_checkpointing=args.gradient_checkpointing,
        modelver=args.modelver,
        overwrite_output_dir=args.overwrite_output_dir,
        eval_every=args.eval_every,
        save_every=args.save_every,
        num_workers=args.num_workers,
        compile_mode=args.compile_mode,
        resume=not args.no_resume,
        force_reprepare=args.reprepare,
        epicycle=epi,
        watermark=watermark,
        loss_guard=not getattr(args, "no_loss_guard", False),
        resume_from=getattr(args, "resume_from", "auto"),
        snapshot_every=getattr(args, "snapshot_every", 200),
        stop_on_divergence=getattr(args, "stop_on_divergence", False),
        loss_chunk_size=getattr(args, "loss_chunk_size", None),
    )


def _run_sft(args: argparse.Namespace, watermark=None, drop_watermark: bool = False) -> None:
    from geocentric.train_sft import sft
    from geocentric.watermark import WatermarkConfig, resolve_watermark

    if watermark is None and not drop_watermark and not getattr(args, "_watermark_resolved", False):
        existing = WatermarkConfig.load(args.model_dir)
        watermark, drop_watermark = resolve_watermark(
            args, args.modelver,
            context="fine-tuning inherits the pretrained checkpoint's watermark; change or "
                    "remove it here if this is a different model to the world",
            existing=existing,
        )
        if existing is not None and watermark is existing:
            watermark = None  # unchanged: let the checkpoint's own config carry it

    sft(
        model_dir=args.model_dir,
        sft_data_path=args.sft_data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size or 8,
        gradient_accumulation_steps=args.gradient_accumulation_steps or 4,
        learning_rate=args.learning_rate or 1e-4,
        warmup_ratio=args.warmup_ratio,
        dtype_name=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        modelver=args.modelver,
        overwrite_output_dir=args.overwrite_output_dir,
        num_workers=args.num_workers,
        compile_mode=args.compile_mode,
        drop_overlong=not args.keep_overlong,
        watermark=watermark,
        drop_watermark=drop_watermark,
        loss_guard=not getattr(args, "no_loss_guard", False),
        snapshot_every=getattr(args, "snapshot_every", 100),
        loss_chunk_size=getattr(args, "loss_chunk_size", None) or 0,
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    from geocentric.watermark import resolve_watermark

    # Asked once for the whole pipeline. Being prompted twice for the same decision,
    # an hour apart, is how a run ends up half-marked.
    watermark, _ = resolve_watermark(
        args, args.modelver,
        context="this pipeline trains a model from scratch and then instruction-tunes it; "
                "the answer applies to both stages",
    )

    pre_args = argparse.Namespace(**vars(args))
    pre_args._watermark_resolved = True
    pre_args.tokenizer_path = None
    pre_args.eval_every = 500
    pre_args.save_every = 1000
    pre_args.no_resume = False
    pre_args.reprepare = False
    pre_args.dropout = 0.0
    pre_args.warmup_ratio = 0.01
    _run_pretrain(pre_args, watermark=watermark)

    sft_args = argparse.Namespace(**vars(args))
    sft_args._watermark_resolved = True
    sft_args.model_dir = args.output_dir
    sft_args.output_dir = args.output_dir
    sft_args.epochs = args.sft_epochs
    sft_args.warmup_ratio = 0.03
    sft_args.keep_overlong = False
    # Pretraining and fine-tuning want different rates — roughly 6e-4 against random
    # weights, 1e-4 against a trained one. Carrying --learning_rate across would
    # silently fine-tune at the pretraining rate and wreck the checkpoint.
    sft_args.learning_rate = args.sft_learning_rate
    # Accumulation is sized for packed pretraining windows; SFT batches are shorter.
    sft_args.gradient_accumulation_steps = None
    # SFT runs are short, so snapshot more often — but 0 means "off" and must stay off.
    pretrain_snapshot = getattr(args, "snapshot_every", 200)
    sft_args.snapshot_every = 0 if pretrain_snapshot == 0 else min(100, pretrain_snapshot)
    # The pretrained checkpoint already carries the mark; passing it again would only
    # rewrite the same file.
    _run_sft(sft_args, watermark=None)


def _run_chat(args: argparse.Namespace) -> None:
    """Interactive test harness that adapts to what the checkpoint actually is.

    A pretrained-only checkpoint has never seen a chat template or a system
    prompt. Feeding it one produces role tags it has no idea how to close, which
    reads as the model being broken when it is simply being asked the wrong kind
    of question. So a base model is driven as a text continuer and only an
    instruction-tuned one gets the chat framing.
    """
    import torch
    from geocentric.device import resolve_dtype
    from geocentric.chat import DEFAULT_SYSTEM
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.generate import build_chat_prompt, stream_text

    try:
        # Gives input() arrow-key history and line editing. Without it an up-arrow
        # arrives as the literal escape sequence and gets fed to the model as text.
        import readline  # noqa: F401
    except ImportError:
        pass

    model, tokenizer, stage = load_model_and_tokenizer(args.model_dir, checkpoint_name=args.checkpoint, with_stage=True)
    device = next(model.parameters()).device
    dtype = resolve_dtype(device, args.dtype)
    if device.type == "cpu" and args.dtype == "auto":
        dtype = torch.float32
    mode = args.mode if args.mode != "auto" else ("chat" if stage in ("sft", "vision") else "base")
    images, image_prefix = _load_chat_image(model, args.image)

    print(BANNER)
    print(f"{model.config.model_name} | {model.num_params():,} params | ctx {model.config.block_size}")
    print(f"checkpoint: {stage}  ->  {mode} mode")
    print()
    if model.config.watermark:
        print(f"  output is watermarked as {model.config.watermark.get('identity')!r}")
    if images is not None:
        print(f"  image attached: {args.image}")
    if mode == "base":
        print("  Base model: no instruction tuning yet, so there is no system prompt and")
        print("  no chat roles. Type the start of a passage and it continues the text.")
        print("  Try:  The capital of France is")
    else:
        print("  Instruction-tuned: answers turns and stops at the end of its own.")
    print("  /reset clears history, /system <text> sets the system prompt, /exit quits.\n")

    system = args.system or DEFAULT_SYSTEM
    history: list[dict[str, str]] = []
    sampling = dict(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )

    prompt_label = "text > " if mode == "base" else "you  > "
    reply_label = "cont > " if mode == "base" else "bot  > "

    while True:
        try:
            line = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break
        if line in {"/reset", "/clear", "/new"}:
            history = []
            print("History cleared.\n")
            continue
        if line == "/help":
            print("  /reset  clear history    /system <text>  set system prompt    /exit  quit\n")
            continue
        if line.startswith("/") and not line.startswith("/system"):
            # Silently generating from a mistyped command looks like the model
            # ignoring you; say so instead.
            print(f"Unknown command {line.split()[0]!r}. Try /help, or prefix with a space "
                  "to send it as text.\n")
            continue
        if line.startswith("/system "):
            system = line[len("/system "):].strip()
            history = []
            print("System prompt set; history cleared.\n" if mode == "chat"
                  else "Base models ignore the system prompt.\n")
            continue

        if mode == "chat":
            # The placeholders go on the first turn only: after that the picture is
            # in the KV cache and in the conversation history.
            content = f"{image_prefix}{line}" if (images is not None and not history) else line
            history.append({"role": "user", "content": content})
            prompt = build_chat_prompt(history, system=system)
        else:
            # Feed the raw text back so the model continues rather than answers.
            prompt = f"{image_prefix}{line}" if images is not None else line

        print(reply_label, end="", flush=True)
        chunks: list[str] = []
        # Passed on every turn, not just the first: the placeholders stay in the
        # rendered history, so the splice has to keep having something to put there.
        try:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                for piece in stream_text(model, tokenizer, prompt, images=images, **sampling):
                    chunks.append(piece)
                    print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print("  [interrupted]", end="")
        print("\n")

        if mode == "chat":
            history.append({"role": "assistant", "content": "".join(chunks).strip()})


def _run_generate(args: argparse.Namespace) -> None:
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.generate import build_chat_prompt, generate_text

    model, tokenizer, stage = load_model_and_tokenizer(args.model_dir, with_stage=True)
    images, image_prefix = _load_chat_image(model, args.image)
    # Same reasoning as chat: only an instruction-tuned checkpoint gets the template.
    use_raw = args.raw or stage not in ("sft", "vision")
    text = f"{image_prefix}{args.prompt}" if images is not None else args.prompt
    prompt = text if use_raw else build_chat_prompt([{"role": "user", "content": text}])
    print(generate_text(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        images=images,
    ))


def _load_chat_image(model, path):
    """Return (image batch, placeholder prefix) for an optional --image."""
    if not path:
        return None, ""
    if not model.config.vision:
        print(f"Ignoring --image: {model.config.model_name} has no vision tower. "
              "Train one with `geocentric train-vision`.")
        return None, ""

    from geocentric.vision import VisionConfig, image_placeholder, load_image

    vision = VisionConfig.from_dict(model.config.vision)
    device = next(model.parameters()).device
    tensor = load_image(path, vision.image_size).unsqueeze(0).to(device)
    return tensor, image_placeholder(vision) + "\n"


def _run_train_vision(args: argparse.Namespace) -> None:
    from geocentric.train_vision import train_vision
    from geocentric.watermark import WatermarkConfig, resolve_watermark

    watermark, _ = resolve_watermark(
        args, args.modelver,
        context="adding vision does not change who made the model, but it is a new "
                "checkpoint and a new chance to set the attribution",
        existing=WatermarkConfig.load(args.model_dir),
    )
    train_vision(
        model_dir=args.model_dir,
        vision_data_path=args.vision_data_path,
        output_dir=args.output_dir,
        image_root=args.image_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        projector_lr_multiplier=args.projector_lr_multiplier,
        image_size=args.image_size,
        patch_size=args.patch_size,
        vision_layers=args.vision_layers,
        vision_width=args.vision_width,
        vision_pool=args.vision_pool,
        loss_chunk_size=args.loss_chunk_size,
        freeze_lm=args.freeze_lm,
        dtype_name=args.dtype,
        modelver=args.modelver,
        num_workers=args.num_workers,
        watermark=watermark,
    )


def _run_bench(args: argparse.Namespace) -> None:
    from geocentric.parallax import run_parallax, write_report

    result = run_parallax(
        args.model_dir, eval_text=args.eval_text, vision_data=args.vision_data,
        only=args.only, checkpoint_name=args.checkpoint, nadir_tokens=args.nadir_tokens,
    )
    output = Path(args.output) if args.output else Path(args.model_dir) / "PARALLAX.md"
    path = write_report(result, output)

    if args.quiet:
        print(path)
        return

    print()
    print(f"  PARALLAX {result.version} — {result.model_name} ({result.stage})")
    print(f"  {'-' * 62}")
    for probe in result.probes:
        score = f"{probe.score:5.1f}" if probe.score is not None else "    —"
        bar = "" if probe.score is None else (
            "█" * int(round(20 * probe.score / 100)) + "░" * (20 - int(round(20 * probe.score / 100)))
        )
        print(f"  {probe.name:<10} {score}  {bar:<20}  {probe.headline}")
    print(f"  {'-' * 62}")
    print(f"  Parallax Index: {result.index:.1f} / 100")
    print(f"\n  Report: {path}")
    print(f"  Data:   {path.with_suffix('.json')}")


def _run_release(args: argparse.Namespace) -> None:
    from geocentric.release import release
    from geocentric.watermark import WatermarkConfig, resolve_watermark

    existing = WatermarkConfig.load(args.model_dir)
    default_identity = existing.identity if existing else Path(args.model_dir).resolve().name
    watermark, removed = resolve_watermark(
        args, default_identity,
        context="this is the last point at which attribution can be added. Once the "
                "weights are published, output from them can never be marked.",
        existing=existing,
    )
    release(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        watermark=watermark,
        remove_watermark=removed or bool(args.no_watermark),
        checkpoint_name=args.checkpoint,
        license_name=args.license_name,
        description=args.description,
        benchmark=not args.no_benchmark,
        eval_text=args.eval_text,
        vision_data=args.vision_data,
        overwrite=args.overwrite,
    )


def _run_detect(args: argparse.Namespace) -> None:
    from geocentric.checkpoint import find_tokenizer_path
    from geocentric.tokenizer_train import load_tokenizer
    from geocentric.watermark import MIN_SCORED_TOKENS, WatermarkConfig, detect_best

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    if not text.strip():
        print("Nothing to test. Pass --text, --file, or pipe text in on stdin.")
        sys.exit(1)

    identities = list(args.identity or [])
    if not identities:
        existing = WatermarkConfig.load(args.model_dir)
        if existing is None:
            print(f"No --identity given and no watermark.json in {args.model_dir}. "
                  "Pass --identity with the name(s) to test against.")
            sys.exit(1)
        identities = [existing.identity]
        args.gamma, args.delta = existing.gamma, existing.delta

    tokenizer = load_tokenizer(find_tokenizer_path(args.model_dir))
    candidates = [WatermarkConfig(identity=name, gamma=args.gamma, delta=args.delta)
                  for name in identities]
    results = detect_best(text, tokenizer, candidates, z_threshold=args.z_threshold)

    print()
    print(f"  {'identity':<34} {'z':>7} {'p':>10}  {'green':>13}")
    print(f"  {'-' * 68}")
    for r in results:
        flag = "  <-- watermarked" if r.watermarked else ""
        print(f"  {r.identity[:34]:<34} {r.z_score:7.2f} {r.p_value:10.3g}  "
              f"{r.green_tokens:5d}/{r.scored_tokens:<5d}{flag}")
    print()

    best = results[0]
    if best.scored_tokens < MIN_SCORED_TOKENS:
        print(f"  UNDECIDABLE — {best.scored_tokens} scored tokens, the test needs at least "
              f"{MIN_SCORED_TOKENS}.")
    elif best.watermarked:
        print(f"  WATERMARKED as {best.identity!r} (z={best.z_score:.2f}, p={best.p_value:.3g}).")
        if len(results) > 1:
            print(f"  Next closest identity scored z={results[1].z_score:.2f}, which is noise.")
    else:
        print(f"  No watermark detected under {'this identity' if len(results) == 1 else 'any of these identities'} "
              f"(best z={best.z_score:.2f}, threshold {args.z_threshold}).")
        print("  A z near zero means either no watermark, a different identity, text that has "
              "been rewritten — or text the model was too certain about to mark.")
        _explain_negative(args, text, candidates[0])
    sys.exit(0 if best.watermarked else 2)


def _explain_negative(args, text: str, config) -> None:
    """Tell a negative result apart from a mark that never had room to exist.

    Reporting "not watermarked" for text a watermarked model genuinely produced is the
    worst failure this tool can have, and it happens whenever the model was certain
    about most of its tokens. It costs one forward pass to say which case this is.
    """
    if args.no_capacity:
        return
    try:
        from geocentric.checkpoint import load_model_and_tokenizer
        from geocentric.watermark import watermark_capacity

        model, tokenizer = load_model_and_tokenizer(args.model_dir)
    except Exception as exc:
        print(f"  (capacity check skipped: {exc})")
        return
    if not model.config.watermark:
        print(f"  The model in {args.model_dir} is not watermarked either, so text it "
              "generated was never marked.")
        return

    report = watermark_capacity(model, tokenizer, text, config)
    print()
    print(f"  Capacity of this model on this text ({report.scored_positions} positions):")
    print(f"    influenceable positions   {report.influenceable_fraction:.1%}")
    print(f"    median predictive entropy {report.median_entropy:.4f} nats")
    print(f"    expected z per sqrt(token) {report.expected_z_per_sqrt_token:.4f}")
    if report.tokens_for_decisive_z:
        print(f"    tokens needed for z=4     ~{report.tokens_for_decisive_z:,}")
    print(f"  {report.verdict(config.gamma)}")


def _run_plan(args: argparse.Namespace) -> None:
    from geocentric.param_compiler import compute_fluid_dimensions, recommended_tokens

    dims = compute_fluid_dimensions(args.preset, vocab_size=args.vocab_size, block_size=args.block_size)
    budget = recommended_tokens(dims["params"])
    print(json.dumps(dims, indent=2))
    print(f"\nParameters:        {dims['params']:,}")
    print(f"Context length:    {dims['block_size']:,} tokens")
    print(f"Token budget:      {budget:,} ({budget / 1e9:.2f}B) for compute-optimal training")
    print(f"Wikipedia (en) is roughly 4B tokens, so this needs about {budget / 4e9:.1f}x English Wikipedia.")


def _run_list_models() -> None:
    roots = [Path("runs"), Path("models")]
    found = False
    for root in roots:
        if not root.exists():
            continue
        for ckpt in sorted(root.rglob("*.pt")):
            size = ckpt.stat().st_size / 1024**2
            print(f"{ckpt}  ({size:,.0f} MB)")
            found = True
    if not found:
        print("No checkpoints found under runs/ or models/.")


def _run_script(filename: str, function: str) -> None:
    """Load a helper from scripts/ by path — it ships beside the package, not inside it."""
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / filename
    if not script.exists():
        print(f"Helper script not found: {script}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getattr(module, function)()


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train-tokenizer":
        from geocentric.data import iter_documents
        from geocentric.tokenizer_train import train_byte_bpe_tokenizer

        train_byte_bpe_tokenizer(
            iter_documents(args.data_path, args.doc_sep), args.output, vocab_size=args.vocab_size
        )
        print(f"Tokenizer saved to {args.output}")
        return

    if args.command == "prepare":
        from geocentric.checkpoint import find_tokenizer_path
        from geocentric.data import prepare_corpus
        from geocentric.tokenizer_train import load_tokenizer

        tok_path = args.tokenizer or find_tokenizer_path(args.output_dir)
        prepare_corpus(
            load_tokenizer(tok_path), args.data_path, Path(args.output_dir) / "corpus",
            val_fraction=args.val_fraction, doc_sep=args.doc_sep,
        )
        return

    if args.command == "pretrain":
        _run_pretrain(args)
    elif args.command == "align-safety":
        from geocentric.alignment import align_safety
        align_safety(args.model_dir, args.output_dir, args.data_path,
                     checkpoint=args.checkpoint, epochs=args.epochs, batch_size=args.batch_size,
                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                     learning_rate=args.learning_rate, dtype_name=args.dtype, yes=args.yes)
    elif args.command == "check-behavior":
        from geocentric.behavior_eval import check_behavior
        check_behavior(args.model_dir, args.data_path, args.output,
                       compare_dir=args.compare_dir, max_new_tokens=args.max_new_tokens)
    elif args.command == "sft":
        _run_sft(args)
    elif args.command == "pipeline":
        _run_pipeline(args)
    elif args.command == "train-vision":
        _run_train_vision(args)
    elif args.command in ("bench", "parallax"):
        _run_bench(args)
    elif args.command == "release":
        _run_release(args)
    elif args.command == "detect":
        _run_detect(args)
    elif args.command in ("chat", "try"):
        if not args.terminal and (args.command == "try" or args.web):
            from geocentric.web_server import serve
            serve(args)
        else:
            _run_chat(args)
    elif args.command == "generate":
        _run_generate(args)
    elif args.command == "plan":
        _run_plan(args)
    elif args.command == "list-models":
        _run_list_models()
    elif args.command == "download-wiki":
        _run_script("download_wikipedia.py", "download_wikitext103")
    elif args.command == "download-alpaca":
        _run_script("download_alpaca.py", "download_alpaca")
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
