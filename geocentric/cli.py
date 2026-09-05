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


def _add_common_training_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~30%% speed for a large memory saving. Use when you hit OOM.")
    p.add_argument("--compile", dest="compile_mode", default="auto", choices=["auto", "off"])
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
    _add_common_training_flags(pre)

    ft = sub.add_parser("sft", help="Instruction fine-tune a pretrained checkpoint")
    ft.add_argument("--model_dir", default="runs/geocentric")
    ft.add_argument("--sft_data_path", required=True)
    ft.add_argument("--output_dir", default=None)
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--warmup_ratio", type=float, default=0.03)
    ft.add_argument("--keep_overlong", action="store_true",
                    help="Truncate conversations longer than the context instead of dropping them")
    _add_common_training_flags(ft)

    pl = sub.add_parser("pipeline", help="Pretrain then SFT in one command")
    pl.add_argument("--data_path", required=True)
    pl.add_argument("--sft_data_path", required=True)
    pl.add_argument("--output_dir", default="runs/geocentric")
    pl.add_argument("--preset", default="120m")
    pl.add_argument("--vocab_size", type=int, default=32000)
    pl.add_argument("--block_size", type=int, default=None)
    pl.add_argument("--epochs", type=int, default=1)
    pl.add_argument("--sft_epochs", type=int, default=3)
    pl.add_argument("--max_steps", type=int, default=0)
    pl.add_argument("--doc_sep", default=None)
    _add_common_training_flags(pl)

    ch = sub.add_parser("chat", help="Chat with a trained checkpoint")
    ch.add_argument("--model_dir", default="runs/geocentric")
    ch.add_argument("--max_new_tokens", type=int, default=256)
    ch.add_argument("--temperature", type=float, default=0.8)
    ch.add_argument("--top_k", type=int, default=50)
    ch.add_argument("--top_p", type=float, default=0.95)
    ch.add_argument("--min_p", type=float, default=0.05)
    ch.add_argument("--repetition_penalty", type=float, default=1.1)
    ch.add_argument("--system", default=None)

    gen = sub.add_parser("generate", help="Generate a single completion and exit")
    gen.add_argument("--model_dir", default="runs/geocentric")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max_new_tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--raw", action="store_true", help="Skip the chat template (base-model completion)")

    pc = sub.add_parser("plan", help="Show the architecture and token budget for a parameter target")
    pc.add_argument("--preset", default="120m")
    pc.add_argument("--vocab_size", type=int, default=32000)
    pc.add_argument("--block_size", type=int, default=None)

    sub.add_parser("download-wiki", help="Download WikiText-103 pretraining data")
    sub.add_parser("download-alpaca", help="Download a cleaned Alpaca instruction set")
    sub.add_parser("list-models", help="List local checkpoints under runs/ and models/")

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
    """Pick a micro-batch that fits VRAM, keeping the tokens-per-step target intact."""
    import torch

    from geocentric.trainer import estimate_batch_size

    if args.batch_size and args.gradient_accumulation_steps:
        return args.batch_size, args.gradient_accumulation_steps

    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    batch = args.batch_size or estimate_batch_size(
        dims["params"], dims["block_size"], vram, args.gradient_checkpointing
    )
    # Aim for roughly half a million tokens per optimizer step, the range small
    # models train most stably in.
    target_tokens = 500_000
    accum = args.gradient_accumulation_steps or max(1, round(target_tokens / (batch * dims["block_size"])))
    return batch, accum


def _run_pretrain(args: argparse.Namespace) -> None:
    from geocentric.train_pretrain import pretrain

    dims = _dims_from_args(args)
    batch, accum = _auto_batch(args, dims)
    print(f"Architecture: {dims['n_layer']}L x {dims['n_embd']}d x {dims['n_head']}h "
          f"(kv {dims['n_kv_head']}) ctx {dims['block_size']} ≈ {dims['params']:,} params")

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
    )


def _run_sft(args: argparse.Namespace) -> None:
    from geocentric.train_sft import sft

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
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    pre_args = argparse.Namespace(**vars(args))
    pre_args.tokenizer_path = None
    pre_args.eval_every = 500
    pre_args.save_every = 1000
    pre_args.no_resume = False
    pre_args.reprepare = False
    pre_args.dropout = 0.0
    pre_args.warmup_ratio = 0.01
    _run_pretrain(pre_args)

    sft_args = argparse.Namespace(**vars(args))
    sft_args.model_dir = args.output_dir
    sft_args.output_dir = args.output_dir
    sft_args.epochs = args.sft_epochs
    sft_args.warmup_ratio = 0.03
    sft_args.keep_overlong = False
    _run_sft(sft_args)


def _run_chat(args: argparse.Namespace) -> None:
    from geocentric.chat import DEFAULT_SYSTEM
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.generate import build_chat_prompt, stream_text

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    print(BANNER)
    print(f"{model.config.model_name} | {model.num_params():,} params | ctx {model.config.block_size}")
    print("Type /reset to clear history, /exit to quit.\n")

    system = args.system or DEFAULT_SYSTEM
    history: list[dict[str, str]] = []

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history = []
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user})
        prompt = build_chat_prompt(history, system=system)
        print("bot > ", end="", flush=True)
        chunks: list[str] = []
        for piece in stream_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        ):
            chunks.append(piece)
            print(piece, end="", flush=True)
        print("\n")
        history.append({"role": "assistant", "content": "".join(chunks).strip()})


def _run_generate(args: argparse.Namespace) -> None:
    from geocentric.checkpoint import load_model_and_tokenizer
    from geocentric.generate import build_chat_prompt, generate_text

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    prompt = args.prompt if args.raw else build_chat_prompt([{"role": "user", "content": args.prompt}])
    print(generate_text(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ))


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
    elif args.command == "sft":
        _run_sft(args)
    elif args.command == "pipeline":
        _run_pipeline(args)
    elif args.command == "chat":
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
