"""Turn a PARALLAX run into a Markdown report that says *why*, not just *what*.

A score on its own tells you nothing you can act on. "SEXTANT 12/100" is only useful
next to "you trained on 8% of the compute-optimal token budget, so the model has not
read enough text to know things yet, and no architecture change will fix that." The
diagnosis section below is a set of explicit rules that read the score alongside
`training_metrics.json` and name the most likely cause. They are heuristics and they
say so; each one states the evidence it fired on so you can disagree with it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geocentric.parallax.suite import ProbeResult, SuiteResult

BAR_WIDTH = 24

PROBE_BLURB = {
    "ZENITH": "how well the model predicts held-out text, in bits per byte",
    "MERIDIAN": "whether a longer context actually improves its predictions",
    "SEXTANT": "whether it ranks a true completion above three false ones",
    "ASTROLABE": "whether it does what an instruction asked, checked by rule",
    "NADIR": "how far it degenerates when generating unaided",
    "LODESTAR": "what the watermark buys and what it costs (reported, not scored)",
    "ORBIT": "throughput and memory (reported, not scored)",
    "PRISM": "whether a multimodal model's answer depends on the image",
}


def _bar(score: Optional[float]) -> str:
    if score is None:
        return "—"
    filled = int(round(BAR_WIDTH * max(0.0, min(100.0, score)) / 100))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _grade(score: float) -> str:
    for cutoff, label in ((80, "strong"), (60, "solid"), (40, "workable"), (20, "weak"), (0, "failing")):
        if score >= cutoff:
            return label
    return "failing"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _epicycle_gears(epicycle: Optional[Dict[str, Any]]) -> str:
    if not epicycle or not epicycle.get("enabled"):
        return "off"
    gears = [g for g in ("deferent", "horizon", "equant", "armillary") if epicycle.get(g)]
    return ", ".join(gears) or "enabled, no gears"


def _loss_slope(history: Dict[str, Any]) -> Optional[float]:
    """Average per-step change in loss over the last fifth of the run.

    A run that ends with the loss still falling steeply did not converge, it ran out
    of scheduled steps. That is the single most common reason a model scores badly,
    and it is invisible in the final loss number alone.
    """
    losses = history.get("loss") or []
    steps = history.get("steps") or []
    if len(losses) < 10 or len(steps) < 10:
        return None
    tail = max(5, len(losses) // 5)
    y0, y1 = losses[-tail], losses[-1]
    x0, x1 = steps[-tail], steps[-1]
    if x1 <= x0:
        return None
    return (y1 - y0) / (x1 - x0)


def diagnose(result: SuiteResult) -> List[Dict[str, str]]:
    """Rule-based causes, each carrying the evidence that triggered it."""
    findings: List[Dict[str, str]] = []
    train = result.training or {}
    config = train.get("config") or {}
    scores = {p.name: p.score for p in result.probes if p.ran}
    detail = {p.name: p.detail for p in result.probes}

    def add(severity: str, title: str, evidence: str, meaning: str, fix: str) -> None:
        findings.append({"severity": severity, "title": title, "evidence": evidence,
                         "meaning": meaning, "fix": fix})

    # --- token budget: the dominant cause at this scale ---------------------
    recommended = config.get("recommended_tokens") or result.params * 20
    seen = train.get("tokens_seen") or (
        (config.get("corpus_tokens") or 0) * max(1, config.get("epochs") or 1)
    )
    if seen and recommended:
        ratio = recommended / max(1, seen)
        if ratio >= 2:
            add(
                "critical" if ratio >= 8 else "major",
                "Trained far below the compute-optimal token budget",
                f"saw {_fmt_int(seen)} tokens against a Chinchilla-style budget of "
                f"{_fmt_int(recommended)} for {_fmt_int(result.params)} parameters — {ratio:.1f}x short",
                "A model that has not read enough text is fluent before it is knowledgeable. "
                "This depresses SEXTANT first, then ZENITH, and leaves generations that are "
                "locally well-formed and factually empty. It is not an architecture problem "
                "and no hyperparameter will fix it.",
                f"add roughly {_fmt_int(max(0, recommended - seen))} more tokens of data, or "
                f"drop to a smaller --preset so the budget matches the corpus you have",
            )
        elif ratio <= 0.3:
            add(
                "minor",
                "Over-trained for this parameter count",
                f"saw {_fmt_int(seen)} tokens against a budget of {_fmt_int(recommended)} — "
                f"{1 / ratio:.1f}x over",
                "Extra tokens past a few times the compute-optimal budget buy very little. "
                "The model is capacity-limited, not data-limited.",
                "spend the next run on a larger --preset rather than more epochs",
            )

    # --- convergence --------------------------------------------------------
    slope = _loss_slope(train.get("history") or {})
    if slope is not None and slope < -1e-4:
        add(
            "major",
            "The run ended while the loss was still falling",
            f"loss fell {abs(slope) * 1000:.3f} per 1,000 steps across the final fifth of training",
            "Training stopped because the step schedule ran out, not because the model "
            "converged. Every score here is a lower bound on what these weights would reach.",
            "raise --epochs or --max_steps and resume; the checkpoint carries optimizer state, "
            "so rerunning the same command continues rather than restarts",
        )

    # --- generalization -----------------------------------------------------
    final_loss = train.get("loss")
    eval_loss = train.get("eval_loss")
    if isinstance(final_loss, (int, float)) and isinstance(eval_loss, (int, float)):
        gap = eval_loss - final_loss
        if gap > 0.3:
            add(
                "major",
                "Validation loss is well above training loss",
                f"train {final_loss:.3f} vs held-out {eval_loss:.3f} (gap {gap:.3f} nats)",
                "The model is memorizing its corpus rather than generalizing, which is why "
                "ZENITH on unseen text is worse than the training curve suggests.",
                "more data first; failing that, raise --dropout or cut --epochs",
            )

    # --- context ------------------------------------------------------------
    meridian = scores.get("MERIDIAN")
    if meridian is not None and meridian < 25:
        gain = (detail.get("MERIDIAN") or {}).get("gain_nats", 0.0)
        block = config.get("block_size") or result.block_size
        add(
            "critical" if gain is not None and gain <= 0 else "major",
            "The model barely benefits from its own context",
            f"loss changes only {gain:+.3f} nats between the start and end of a "
            f"{block}-token window",
            "Predictions late in a document should be easier than predictions at its start. "
            "When they are not, the model has learned local statistics and nothing longer. "
            "The usual cause is a corpus fed in as fragments — a document separator that "
            "matches too often, or records that are each a single sentence.",
            "check --doc_sep against your data: plain text with no separator is treated as "
            "one continuous document, which is usually what you want. Verify with "
            "`geocentric prepare` that the token count per document is in the thousands",
        )

    # --- vocabulary and context width --------------------------------------
    if result.vocab_size and result.vocab_size < 16000:
        add(
            "major",
            "Small vocabulary",
            f"{_fmt_int(result.vocab_size)} tokens",
            "Below about 16k the tokenizer emits sub-word rubble, so the model spends "
            "capacity relearning spelling and its effective context in characters is a "
            "fraction of its block size.",
            "retrain the tokenizer with --vocab_size 32000",
        )
    if result.block_size < 512:
        add(
            "major",
            "Short context",
            f"block_size {result.block_size}",
            "A few hundred tokens is not enough span for a model to learn to hold a topic, "
            "which caps MERIDIAN no matter how long you train.",
            "retrain with --block_size 1024 or higher",
        )

    # --- degeneracy ---------------------------------------------------------
    nadir = detail.get("NADIR") or {}
    loops = nadir.get("looped_generations", 0)
    if loops:
        if result.stage == "pretrained":
            add(
                "minor",
                "Repetition loops in raw generation",
                f"{loops} of {len(nadir.get('samples') or [])} samples looped with the "
                "repetition penalty disabled",
                "Base models loop; this probe runs without a repetition penalty on purpose, "
                "so some looping here is expected and the chat path masks it.",
                "no action unless it persists after SFT — `chat` applies "
                "--repetition_penalty 1.1 by default",
            )
        else:
            add(
                "major",
                "An instruction-tuned model is still looping",
                f"{loops} generation(s) fell into a repeat cycle",
                "After SFT this usually means the end-of-turn token is undertrained, so the "
                "model never learns that finishing is an option.",
                "confirm SFT ran long enough (`total_steps` below), and that overlong "
                "conversations were dropped rather than truncated — truncation removes the "
                "stop token from exactly the examples that teach stopping",
            )

    # --- SFT specifics ------------------------------------------------------
    if result.stage in {"sft", "vision"}:
        lr = config.get("learning_rate")
        if isinstance(lr, (int, float)) and lr < 3e-5:
            add(
                "critical",
                "Fine-tuning rate far too low for a from-scratch model",
                f"learning_rate {lr:.1e}",
                "2e-5 is a rate for a fully-trained multi-billion-parameter model. On a small "
                "model pretrained from scratch it barely moves the weights, so instruction "
                "following never takes hold and the model echoes prompts back.",
                "rerun SFT at 1e-4",
            )
        steps = config.get("total_steps") or 0
        if steps and steps < 100:
            add("major", "Very short fine-tune", f"{_fmt_int(steps)} optimizer steps",
                "Format compliance is the first thing SFT teaches and it still needs a few "
                "hundred steps. ASTROLABE will be dominated by this.",
                "raise --sft_epochs, or use a larger instruction set")
        dropped = config.get("dropped_overlong") or 0
        examples = config.get("examples") or 0
        if examples and dropped / max(1, examples + dropped) > 0.2:
            share = dropped / max(1, examples + dropped)
            add("major", "A large share of the instruction data was dropped as overlong",
                f"{_fmt_int(dropped)} of {_fmt_int(examples + dropped)} conversations ({share:.0%})",
                "Dropping is the right call — truncating would strip the stop token — but "
                "losing a fifth of the data costs real instruction-following quality, and the "
                "dropped examples are systematically the longest and most detailed ones.",
                "raise --block_size so long conversations fit, or filter the dataset yourself")

    astrolabe = detail.get("ASTROLABE") or {}
    if astrolabe.get("stop_discipline") is not None and astrolabe["stop_discipline"] < 0.5:
        add("major", "The model rarely emits a stop token",
            f"stopped on its own in {astrolabe['stop_discipline']:.0%} of ASTROLABE prompts",
            "`<|eot|>` is supervised during SFT and is what teaches a model to finish a turn. "
            "Low stop discipline reads to a user as the model rambling past its answer.",
            "verify the SFT data renders through the chat template with assistant turns ending "
            "in <|eot|>, and that those tokens are inside the supervised span")

    # --- the tokenizer, not the model ---------------------------------------
    zenith_detail = detail.get("ZENITH") or {}
    unk_rate = zenith_detail.get("unk_rate") or 0.0
    if unk_rate > 0.01:
        add("critical", "The tokenizer cannot represent the evaluation text",
            f"{unk_rate:.1%} of scored tokens came back as <unk>"
            + (f", and the vocabulary is only {result.vocab_size:,} tokens"
               if result.vocab_size < 16000 else ""),
            "An <unk> is a character the tokenizer has no way to encode, so the model is "
            "being asked to predict text it cannot even see. ZENITH and SEXTANT are "
            "measuring the vocabulary here, not the model, and generated text will be "
            "missing characters outright. Fix this before reading any other score.",
            "retrain the tokenizer on a corpus that covers the character set you care "
            "about (`geocentric train-tokenizer --vocab_size 32000`), then retrain the "
            "model — an existing checkpoint is tied to its vocabulary and cannot be "
            "moved to a new one")

    # --- the watermark --------------------------------------------------------
    lodestar = detail.get("LODESTAR") or {}
    if lodestar and lodestar.get("mean_z_true_identity") is None:
        # The probe could not decide. Saying "your watermark is weak" on the strength
        # of a measurement that did not happen would be worse than saying nothing.
        add("info", "The watermark could not be measured on this run",
            f"replies averaged {lodestar.get('mean_generation_tokens', 0):.0f} tokens; "
            f"the z-test needs at least {lodestar.get('minimum_tokens', 40)}",
            "LODESTAR needs a paragraph of output to decide anything. This model stops "
            "sooner than that on the probe prompts, so the mark is neither confirmed nor "
            "ruled out here.",
            "run `geocentric detect` on real generated text of a few hundred tokens, which "
            "also reports how much watermark capacity this model has")
    elif lodestar:
        z = lodestar.get("mean_z_true_identity", 0.0)
        cost = lodestar.get("quality_cost_nats_per_token", 0.0)
        if z < 4:
            add("major", "The watermark is not reliably detectable at normal reply length",
                f"mean z={z:.2f} over {lodestar.get('mean_generation_tokens', 0):.0f}-token "
                f"replies, against a decision threshold of 4",
                "Short outputs from this model cannot be attributed. The mark is still there "
                "and long passages will still test positive, but a one-paragraph answer will "
                "not.",
                f"raise delta above {lodestar.get('delta')} at release time — "
                "`geocentric release --watermark_delta 4.0` — and re-run this probe")
        if cost > 0.15:
            add("major", "The watermark is measurably degrading output",
                f"{cost:.3f} nats/token more surprising to the model than its unmarked "
                "generation from the same seed",
                "The bias is pushing generation far enough off the model's preferred path to "
                "cost real quality. Small models have less headroom for this than large ones, "
                "because their second-choice token is often much worse than their first.",
                f"lower delta below {lodestar.get('delta')}, or accept a lower z. "
                "`geocentric bench --only LODESTAR` re-measures in under a minute")

    # --- domain balance -----------------------------------------------------
    zenith = detail.get("ZENITH") or {}
    if zenith.get("spread", 0) > 0.8:
        add("minor", "Uneven across genres",
            f"{zenith['spread']:.2f} bits/byte between {zenith.get('best_slice')} "
            f"(best) and {zenith.get('worst_slice')} (worst)",
            "The corpus is heavily weighted toward one kind of text. That is a legitimate "
            "choice, but it means the headline ZENITH number does not transfer to the weak "
            "genre.",
            f"mix in more {zenith.get('worst_slice')}-like data if you need it")

    # --- vision -------------------------------------------------------------
    prism = detail.get("PRISM") or {}
    if prism and prism.get("gap_nats") is not None and prism["gap_nats"] < 0.03:
        add("critical", "The vision tower is not grounding the text",
            f"swapping in the wrong image changes the loss by only "
            f"{prism['gap_nats']:+.3f} nats",
            "The model produces caption-shaped text from the prompt alone. Captions will read "
            "plausibly one at a time and be unrelated to the actual picture.",
            "train the projector harder first (--freeze_lm for a stage-1 pass), raise "
            "--projector_lr_multiplier, or check that images are being loaded rather than "
            "silently skipped — see missing_images in training_metrics.json")

    # --- loss stability -----------------------------------------------------
    guard = train.get("loss_guard") or {}
    spikes = guard.get("spikes") or 0
    rollbacks = guard.get("rollbacks") or 0
    if rollbacks:
        add("major", "The run hit loss spikes bad enough to roll back",
            f"{spikes} spike(s), {guard.get('skipped_updates', 0)} update(s) dropped, "
            f"{rollbacks} rollback(s) at step(s) "
            f"{', '.join(f'{s:,}' for s in guard.get('rollback_steps', []))}",
            "Something in the corpus is producing gradients far outside the normal "
            "distribution — usually a run of corrupt text, a very long unbroken line, or "
            "binary that survived into a .txt. The guard recovered, but the run lost the "
            "steps between each spike and its last snapshot.",
            "find the offending data: the spike steps above map to positions in the shuffled "
            "corpus, and `--snapshot_every` smaller costs RAM but loses less on each "
            "recovery. If it recurs at the same loss level rather than the same step, lower "
            "--learning_rate instead")
    elif spikes > 5:
        add("minor", "Occasional loss spikes, all absorbed",
            f"{spikes} spike(s), {guard.get('skipped_updates', 0)} update(s) dropped, no rollback",
            "The guard dropped these before they reached the weights, so they cost a handful "
            "of updates and nothing else. Worth knowing about, not worth acting on.",
            "no action; run with --no_loss_guard to see what they would have done")
    if guard.get("exhausted_at_step") is not None:
        add("critical", "The loss guard used up its rollback budget and stood down",
            f"stopped intervening at step {guard['exhausted_at_step']:,} after "
            f"{rollbacks} rollback(s) failed to settle the run",
            "The spikes kept coming after every recovery, so the guard let the updates "
            "through rather than stalling the run forever. Everything after that step "
            "trained unprotected, and the weights may carry damage the loss curve alone "
            "will not show.",
            "this is a data or learning-rate problem, not a guard problem: halve "
            "--learning_rate and resume from the best checkpoint "
            "(`--resume_from best`), or find and remove the corrupt region of the corpus")
    if guard.get("plateau_since_step") is not None:
        add("major", "The loss stopped improving well before the run ended",
            f"no improvement since step {guard['plateau_since_step']:,} "
            f"(best {guard.get('best_loss')} at step {guard.get('best_step', 0):,})",
            "The remaining steps bought nothing measurable. Either the learning rate had "
            "already decayed too far to move the weights, or the model has extracted what "
            "this corpus has to give at this parameter count.",
            "more data, or a larger --preset. Rerunning the same configuration for longer "
            "will reproduce this plateau")

    # --- efficiency ---------------------------------------------------------
    mfu = train.get("mfu")
    if isinstance(mfu, (int, float)) and 0 < mfu < 0.15:
        add("minor", "Low hardware utilization during training",
            f"{mfu:.1%} model FLOPs utilization",
            "Most of the run was spent waiting rather than computing — usually the data "
            "loader, a micro-batch too small to fill the device, or eager mode.",
            "raise --batch_size until memory is tight, raise --num_workers, and leave "
            "--compile on")

    epicycle = config.get("epicycle") or {}
    if epicycle.get("enabled"):
        add("info", "Trained with EPICYCLE",
            f"gears: {_epicycle_gears(epicycle)}",
            "Elastic depth and context apply only during training; the checkpoint is an "
            "ordinary full-depth model and every score here was measured at full capacity. "
            "The one thing to watch is that ARMILLARY trades a little optimizer quality for "
            "memory, so compare against a run without it before concluding the gears helped.",
            "`geocentric bench --compare <other_run>` puts two reports side by side")

    if not findings:
        add("info", "Nothing stood out",
            "no diagnostic rule fired",
            "Scores are consistent with the training statistics on record. Either the run is "
            "healthy or the metrics file is missing the fields the rules read.",
            "run with a larger --eval_text for a more discriminating measurement")

    order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    return findings


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _probe_detail_block(probe: ProbeResult) -> str:
    lines = [f"### {probe.name}", "", f"*{PROBE_BLURB.get(probe.name, '')}*", ""]
    if not probe.ran:
        lines += [f"**Not run.** {probe.note}", ""]
        return "\n".join(lines)
    lines += [f"**{probe.headline}**" + (f" — score {probe.score:.1f}/100" if probe.score is not None
                                         else " — not scored"), ""]
    if probe.note:
        lines += [probe.note, ""]

    detail = probe.detail
    if probe.name == "ZENITH":
        lines += ["| slice | bits/byte |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in detail.get("by_slice", {}).items()]
        lines += ["", f"Scored over {_fmt_int(detail.get('bytes_scored'))} bytes.", ""]
    elif probe.name == "MERIDIAN":
        curve = detail.get("curve_nats") or []
        width = detail.get("bucket_width", 0)
        lines += ["| context position | mean nats/token |", "|---|---|"]
        for i, value in enumerate(curve):
            span = f"{i * width}–{(i + 1) * width - 1}"
            lines.append(f"| {span} | {value:.4f} |" if value is not None else f"| {span} | — |")
        lines += ["", "Lower is better, and it should fall down the table.", ""]
    elif probe.name == "SEXTANT":
        lines += [f"- accuracy **{detail.get('accuracy', 0):.1%}** against "
                  f"{detail.get('chance', 0):.0%} chance",
                  f"- mean log-probability margin over the best distractor: "
                  f"{detail.get('mean_margin', 0):+.3f}", ""]
        if detail.get("misses"):
            lines += ["Missed:", ""] + [f"- {m}" for m in detail["misses"]] + [""]
    elif probe.name == "ASTROLABE":
        lines += ["| item | pass | stopped | answer |", "|---|---|---|---|"]
        for row in detail.get("results", []):
            answer = str(row["answer"]).replace("|", "\\|").replace("\n", " ⏎ ")[:90]
            lines.append(f"| `{row['id']}` | {'✅' if row['pass'] else '❌'} | "
                         f"{'✅' if row['stopped'] else '❌'} | {answer or '*(empty)*'} |")
        lines.append("")
    elif probe.name == "NADIR":
        lines += [f"- distinct-1 {detail.get('distinct_1', 0):.3f} | "
                  f"distinct-2 {detail.get('distinct_2', 0):.3f} | "
                  f"distinct-3 {detail.get('distinct_3', 0):.3f}", ""]
        lines += ["| prompt | tokens | distinct-3 | longest loop |", "|---|---|---|---|"]
        for row in detail.get("samples", []):
            lines.append(f"| {row['prompt'][:48]} | {row['tokens']} | "
                         f"{row['distinct_3']:.2f} | {row['longest_loop']}x |")
        lines.append("")
    elif probe.name == "ORBIT":
        lines += [f"- prefill **{detail.get('prefill_tokens_per_second', 0):,.0f}** tok/s",
                  f"- decode **{detail.get('decode_tokens_per_second', 0):,.0f}** tok/s",
                  f"- peak memory {detail.get('peak_memory_gb', 0):.2f} GB on "
                  f"`{detail.get('device')}`", ""]
    elif probe.name == "LODESTAR":
        lines += [
            f"- identity: **{detail.get('identity')}** (gamma {detail.get('gamma')}, "
            f"delta {detail.get('delta')})",
            f"- z under the true identity: **{detail.get('mean_z_true_identity', 0):+.2f}**",
            f"- z under a decoy identity: {detail.get('mean_z_decoy_identity', 0):+.2f} "
            "*(must be near zero, or the mark identifies nobody in particular)*",
            f"- z on the same model's unmarked output: {detail.get('mean_z_unmarked_text', 0):+.2f} "
            "*(the false-positive rate)*",
            f"- quality cost: **{detail.get('quality_cost_nats_per_token', 0):+.4f}** nats/token "
            "against an identically-seeded unmarked generation",
            f"- measured over {detail.get('samples')} replies averaging "
            f"{detail.get('mean_generation_tokens', 0):.0f} tokens",
            "",
            "Every other generation probe in this report runs the model **unmarked**, because "
            "they measure the model. This one measures the mark.",
            "",
        ]
    elif probe.name == "PRISM":
        lines += [f"- loss with the matching image: **{detail.get('matched_loss', 0):.4f}**",
                  f"- loss with a mismatched image: **{detail.get('mismatched_loss', 0):.4f}**",
                  f"- grounding gap: **{detail.get('gap_nats', 0):+.4f}** nats over "
                  f"{detail.get('pairs', 0)} pairs", ""]
    return "\n".join(lines)


def render_report(result: SuiteResult) -> str:
    scored = [p for p in result.probes if p.ran]
    ranked = sorted(scored, key=lambda p: p.score or 0, reverse=True)
    findings = diagnose(result)
    train = result.training or {}
    config = train.get("config") or {}

    out: List[str] = []
    out += [
        f"# PARALLAX report — {result.model_name}",
        "",
        f"`{result.model_dir}` · {result.stage} checkpoint · "
        f"{_fmt_int(result.params)} parameters · generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| | |",
        "|---|---|",
        f"| Parallax Index | **{result.index:.1f} / 100** ({_grade(result.index)}) |",
        f"| Stage | {result.stage} |",
        f"| Parameters | {_fmt_int(result.params)} |",
        f"| Context | {_fmt_int(result.block_size)} tokens |",
        f"| Vocabulary | {_fmt_int(result.vocab_size)} |",
        f"| Multimodal | {'yes' if result.multimodal else 'no'} |",
        f"| Watermark | {result.watermark or 'none'} |",
        f"| Device | `{result.device}` |",
        f"| Suite | PARALLAX {result.version}, {result.seconds:.1f}s |",
        "",
        "## Scores",
        "",
        "| probe | score | | measured |",
        "|---|---:|---|---|",
    ]
    for probe in result.probes:
        score = f"{probe.score:.1f}" if probe.score is not None else "—"
        out.append(f"| **{probe.name}** | {score} | `{_bar(probe.score)}` | {probe.headline} |")
    out += [
        "",
        "The **Parallax Index** is the weighted mean of the scored probes: "
        + ", ".join(f"{k} {v:.0%}" for k, v in result.weights.items())
        + ". Probes that did not run are excluded and the remaining weights renormalized, "
          "so a skipped probe never counts as a zero. ORBIT measures the machine as much as "
          "the model and is deliberately left out.",
        "",
    ]

    # --- strengths / weaknesses ---
    out += ["## Strengths", ""]
    strong = [p for p in ranked if (p.score or 0) >= 40][:3]
    if strong:
        for probe in strong:
            out.append(f"- **{probe.name} — {probe.score:.1f}/100.** {probe.headline}. {probe.note}")
    else:
        out.append("- None of the probes cleared 40/100. The strongest was "
                   f"**{ranked[0].name}** at {ranked[0].score:.1f}."
                   if ranked else "- No probe produced a score.")
    out += ["", "## Weaknesses", ""]
    weak = [p for p in reversed(ranked) if (p.score or 0) < 60][:4]
    if weak:
        for probe in weak:
            out.append(f"- **{probe.name} — {probe.score:.1f}/100.** {probe.headline}. {probe.note}")
    else:
        out.append("- Every probe scored 60 or above.")

    # --- diagnosis ---
    out += ["", "## Why — diagnosis from the training record", ""]
    if not train:
        out += ["No `training_metrics.json` was found next to the checkpoint, so the causes "
                "below are inferred from the scores alone. Benchmark a directory that a "
                "Geocentric run wrote to for the full analysis.", ""]
    icons = {"critical": "🔴", "major": "🟠", "minor": "🟡", "info": "🔵"}
    for finding in findings:
        out += [
            f"### {icons.get(finding['severity'], '·')} {finding['title']}",
            "",
            f"- **Evidence** — {finding['evidence']}",
            f"- **What it means** — {finding['meaning']}",
            f"- **What to do** — {finding['fix']}",
            "",
        ]

    # --- training record ---
    out += ["## Training record", ""]
    if train:
        rows: List[Tuple[str, str]] = [
            ("phase", str(train.get("phase", "—"))),
            ("status", str(train.get("status", "—"))),
            ("steps", _fmt_int(train.get("step"))),
            ("tokens seen", _fmt_int(train.get("tokens_seen"))),
            ("compute-optimal budget", _fmt_int(config.get("recommended_tokens"))),
            ("corpus tokens", _fmt_int(config.get("corpus_tokens"))),
            ("final train loss", f"{train['loss']:.4f}" if isinstance(train.get("loss"), (int, float)) else "—"),
            ("final eval loss", f"{train['eval_loss']:.4f}" if isinstance(train.get("eval_loss"), (int, float)) else "—"),
            ("learning rate", f"{config['learning_rate']:.2e}" if config.get("learning_rate") else "—"),
            ("tokens/step", _fmt_int(config.get("tokens_per_step"))),
            ("dtype", str(config.get("dtype", "—")).replace("torch.", "")),
            # Only report gears that were actually running: a disabled EpicycleConfig
            # still carries each gear's default, and printing those reads as a lie.
            ("EPICYCLE", _epicycle_gears(config.get("epicycle"))),
            ("best loss", str((train.get("loss_guard") or {}).get("best_loss", "—"))),
            ("loss spikes / rollbacks",
             f"{(train.get('loss_guard') or {}).get('spikes', 0)} / "
             f"{(train.get('loss_guard') or {}).get('rollbacks', 0)}"),
        ]
        out += ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows] + [""]
    else:
        out += ["*(no training metrics on file)*", ""]

    # --- probe detail ---
    out += ["## Probe detail", ""]
    out += [_probe_detail_block(p) for p in result.probes]

    out += [
        "## Reproducing this",
        "",
        "```bash",
        f"geocentric bench --model_dir {result.model_dir}",
        "```",
        "",
        "The bundled probes are a few kilobytes of text written for this suite rather than "
        "sampled from a public corpus, so a model trained on Wikipedia has not already read "
        "them. That keeps them honest and keeps them small: treat the numbers as a calibrated "
        "smoke test, not a leaderboard. Pass `--eval_text` with your own held-out data to run "
        "the same probes at whatever size and domain you actually care about.",
        "",
    ]
    return "\n".join(out) + "\n"


def write_report(result: SuiteResult, path: str | Path, also_json: bool = True) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result), encoding="utf-8")
    if also_json:
        target.with_suffix(".json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
    return target
