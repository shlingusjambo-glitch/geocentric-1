"""Train one arm of the A/B comparison and score it. Runs inside a given code tree.

This file is copied into both the legacy worktree and the current tree, so it may
only touch API surface that exists in both. Version differences are handled by
filtering kwargs through inspect.signature.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import torch


def _call_filtered(fn, **kwargs):
    """Pass only the keyword arguments this version of `fn` actually accepts."""
    allowed = set(inspect.signature(fn).parameters)
    dropped = sorted(set(kwargs) - allowed)
    if dropped:
        print(f"[worker] this tree does not support: {', '.join(dropped)}")
    return fn(**{k: v for k, v in kwargs.items() if k in allowed})


def _load_model_and_tokenizer(model_dir: str, device: torch.device):
    from geocentric.checkpoint import load_model_and_tokenizer

    params = inspect.signature(load_model_and_tokenizer).parameters
    kwargs = {}
    if "device" in params:
        kwargs["device"] = device
    if "dtype" in params:
        kwargs["dtype"] = torch.float32
    model, tokenizer = load_model_and_tokenizer(model_dir, **kwargs)
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def bits_per_byte(model, tokenizer, text: str, device: torch.device, stride_fraction: float = 0.5) -> dict:
    """Tokenizer-independent held-out score.

    Cross-entropy per token cannot be compared across models with different
    tokenizers and vocabulary sizes — a 32k vocab spreads the same text over fewer,
    harder-to-predict tokens than an 8k vocab. Normalizing the total negative log
    likelihood by the byte count of the source text removes that dependence, so the
    two arms become directly comparable.

    Every scored token is given exactly `block_size - stride` tokens of context,
    including in the first window. Uniform context matters: if the first window were
    scored from position 1 while later windows carried a full prefix, the arm with
    the shorter context would be credited for a larger share of cheap, low-context
    predictions, and the byte denominator would no longer match the tokens scored.
    """
    block = int(model.config.block_size)
    stride = max(1, int(block * stride_fraction))
    context = block - stride

    encoding = tokenizer.encode(text)
    ids = encoding.ids
    if len(ids) <= block + stride:
        raise ValueError(f"Held-out text is too short: {len(ids)} tokens for block_size {block}.")

    total_nll = 0.0
    scored_tokens = 0
    first_scored = context + 1
    last_scored = first_scored - 1
    start = 0

    while start + block < len(ids):
        window = ids[start : start + block + 1]
        inputs = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([window[1:]], dtype=torch.long, device=device).clone()
        targets[:, :context] = -100

        logits, _ = model(inputs)
        total_nll += float(torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ))
        scored_tokens += int((targets != -100).sum())
        last_scored = start + block
        start += stride

    # Count only the bytes the scored tokens actually cover, so numerator and
    # denominator describe the same span of text.
    span_start = encoding.offsets[first_scored][0]
    span_end = encoding.offsets[last_scored][1]
    scored_bytes = len(text[span_start:span_end].encode("utf-8"))

    return {
        "bits_per_byte": total_nll / math.log(2) / scored_bytes,
        "nats_per_token": total_nll / scored_tokens,
        "token_perplexity": math.exp(total_nll / scored_tokens),
        "scored_tokens": scored_tokens,
        "scored_bytes": scored_bytes,
        "tokens_per_byte": scored_tokens / scored_bytes,
        "context_per_scored_token": context,
    }


@torch.no_grad()
def sample(model, tokenizer, prompts: list[str], device: torch.device, max_new_tokens: int = 60) -> list[dict]:
    out = []
    allowed = set(inspect.signature(model.generate).parameters)
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids[-model.config.block_size // 2 :]
        kwargs = {"max_new_tokens": max_new_tokens, "temperature": 0.8, "top_k": 50}
        eos = tokenizer.token_to_id("<eos>")
        if eos is not None:
            kwargs["eos_id"] = eos
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            generated = model.generate(torch.tensor([ids], dtype=torch.long, device=device), **kwargs)
            text = tokenizer.decode(generated[0, len(ids) :].tolist(), skip_special_tokens=True)
        except Exception as exc:  # a broken arm should not kill the comparison
            text = f"<generation failed: {exc}>"
        out.append({"prompt": prompt, "completion": text})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--heldout_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--preset", default="50m")
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--block_size", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--compile", dest="compile_mode", default="auto")
    ap.add_argument("--heldout_chars", type=int, default=400_000)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from geocentric.param_compiler import compute_fluid_dimensions
    from geocentric.train_pretrain import pretrain

    result: dict = {"arm": args.arm, "tree": os.getcwd()}

    dims = compute_fluid_dimensions(args.preset, vocab_size=args.vocab_size)
    # When the orchestrator supplies dimensions, both arms are built at the same
    # depth and width. The 2.1 planner mis-derives sizes (it deducts the tied
    # embedding matrix twice), so left to itself each arm would build a different
    # sized model and the score difference would mostly measure that.
    for key, value in (("n_layer", args.n_layer), ("n_head", args.n_head), ("n_embd", args.n_embd)):
        if value:
            dims[key] = value
    if args.n_head and dims.get("n_kv_head"):
        dims["n_kv_head"] = max(1, min(dims["n_kv_head"], args.n_head))
        while args.n_head % dims["n_kv_head"]:
            dims["n_kv_head"] -= 1
    result["dims"] = dims
    print(f"[{args.arm}] architecture: {dims}")

    batch = args.batch_size or 8
    accum = args.gradient_accumulation_steps or 8

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _call_filtered(
        pretrain,
        data_path=args.train_file,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=dims["n_layer"],
        n_head=dims["n_head"],
        n_kv_head=dims.get("n_kv_head"),
        n_embd=dims["n_embd"],
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.learning_rate if args.learning_rate else 6e-4,
        dtype_name="auto",
        modelver=f"ab_{args.arm}",
        overwrite_output_dir=True,
        compile_mode=args.compile_mode,
        # The legacy trainer stops early on epoch-level patience; disable so both
        # arms consume the same corpus.
        patience=0,
    )

    result["train_seconds"] = time.perf_counter() - started
    result["peak_host_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    result["peak_vram_gb"] = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )

    metrics_path = Path(args.output_dir) / "training_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["final_train_loss"] = metrics.get("loss")
        result["steps"] = metrics.get("step")
        result["tokens_per_second"] = metrics.get("tokens_per_second")
        result["corpus_tokens"] = metrics.get("config", {}).get("corpus_tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model_and_tokenizer(args.output_dir, device)
    result["params"] = sum(p.numel() for p in model.parameters())
    result["vocab_size_actual"] = int(tokenizer.get_vocab_size())
    result["block_size_actual"] = int(model.config.block_size)

    heldout = Path(args.heldout_file).read_text(encoding="utf-8", errors="replace")[: args.heldout_chars]
    result["heldout"] = bits_per_byte(model, tokenizer, heldout, device)
    result["samples"] = sample(
        model, tokenizer,
        [
            "The history of the Roman Empire",
            "In physics, energy is",
            "The city of Paris is",
        ],
        device,
    )

    Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{args.arm}] bits/byte = {result['heldout']['bits_per_byte']:.4f}")


if __name__ == "__main__":
    main()
