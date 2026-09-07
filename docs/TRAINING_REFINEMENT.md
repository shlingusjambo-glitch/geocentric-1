# Training refinement: selected replay, source losses, optional alignment

This pass extends EPICYCLE without changing model parameter names or shapes. Existing checkpoints remain loadable and trainable. Existing presets keep their previous defaults. No long training run was started as part of this pass.

## EQUANT selected replay

EQUANT already assigns zero gradient to tokens outside its selected difficulty band. The previous chunked vocabulary loss still recomputed those rows during backward. Selected replay scores every row in forward, compacts the nonzero gradient rows once, and recomputes only those vocabulary projections and cross-entropies during backward. Transformer backward still runs normally. This is an implementation optimization of the existing first-order objective, not a new architecture or a claim of a novel research result.

```bash
# New run: same training policy as quality, with selected replay enabled.
geocentric pretrain --data_path data --output_dir runs/selective \
  --preset 120m --epicycle selective --compile off --loss_chunk_size 256

# Existing EQUANT run: keep its preset, optimizer, architecture and other settings.
# Add these options to the original command:
# --equant_sparse_replay --compile off --loss_chunk_size 256
```

The extra flag accepts `quality`, `selective`, or `full`. It does not turn on EQUANT for presets that previously used ordinary cross-entropy. Switching a `capacity` or `balanced` run to `selective` would also change the optimizer; do not do that just to enable this optimization. Keep the original command and preset when resuming.

Dynamic row compaction is currently eager-only. With compilation, the loss uses the established checkpointed implementation and the trainer prints that selected replay is inactive. A zero chunk size also disables this path. Higher-order differentiation is unsupported. Floating-point accumulation order changes, so bit-for-bit training trajectories are not promised. Selecting most rows, a small vocabulary, or synchronization overhead can erase the speed benefit.

### Measured speed

Two matched synthetic full-model AdamW comparisons on the local Apple M4, PyTorch 2.8, MPS BF16, batch 4, context 256, vocabulary 16,000, four layers of width 256:

| Measurement | Checkpointed EQUANT | Selected replay | Throughput increase |
|---|---:|---:|---:|
| 3 warmups, 12 measured updates | 14,037 tokens/s | 14,787 tokens/s | 5.34% |
| 10 warmups, 60 measured updates | 15,349 tokens/s | 16,096 tokens/s | 4.87% |

Each pair starts from identical weights, sees the same batches, alternates execution order, and synchronizes the device around updates. The benchmark uses EQUANT's selected band throughout; real runs only use it after their configured EQUANT start. It excludes dataset loading, validation and saving. This is approximately **5% faster in this benchmark**, not a promised end-to-end gain, a quality improvement, or an RTX 2060 measurement. Raw timings and losses are in `research/benchmarks/sparse-replay-m4*.json`.

The earlier user-reported RTX improvement from 16,000–17,000 to about 20,000 tokens/s was approximately **17.6–25%**. It is a separate, uncontrolled observation, not evidence that the additional 5% measured on M4 will transfer to that machine. No parameter-capacity increase is claimed for this pass.

On the RTX 2060, in the activated environment:

```bash
bash scripts/validate_cuda.sh
python scripts/sparse_replay_bench.py --device cuda --steps 60 --warmup 10 \
  --output runs/cuda-validation/sparse-replay.json
```

The validation script refuses to report success without an actual CUDA device and executes a kernel before tests. Auto precision continues to choose FP16 plus gradient scaling for Turing. This pass was locally tested on CPU/MPS; NVIDIA execution and Ubuntu 26.04 GPU drivers still need verification on the target machine.

## See HTML and language loss separately

New corpus preparation records each source file's token ranges in both split metadata files. At normal validation events, pretraining evaluates up to two evenly spaced windows per source at full model depth, using ordinary cross-entropy. `training_metrics.json` stores `source_eval` and `source_eval_step`; `scripts/watch_training.py` displays each filename, loss, and number of scored tokens. For example, `html.txt` and `language.jsonl` appear separately when they are separate inputs.

These are sampled held-out next-token losses in nats, not accuracy percentages or factuality scores. They do not classify the contents of a mixed file into topics. A source shorter than two validation tokens has no usable score. Short sources use a shorter context; sample counts are shown so a tiny estimate is not mistaken for a full-corpus assessment. Aggregate validation and best-checkpoint selection retain their existing behavior.

Old prepared corpora without source ranges still work, but cannot retrospectively provide source scores. To enable them, explicitly reprepare using the same tokenizer, corpus and split settings; do not change the data of a running experiment unintentionally. New runs produce the metadata automatically. Turning validation off during re-preparation now removes obsolete validation files. Splits are token-disjoint source tails, not guaranteed document-disjoint or duplicate-free datasets.

## Optional post-SFT alignment

```bash
geocentric align-safety --model_dir runs/my-sft \
  --output_dir runs/my-alignment --data_path data/reviewed-alignment.jsonl
```

This is a separate command; pretraining, SFT and the pipeline do not invoke it automatically. It requires an SFT or vision checkpoint. Use `--checkpoint filename.pt` to select one explicitly. The command validates the data and prints the policy, source, output paths, example counts, epochs and learning rate, then asks `Continue with alignment? [y/N]`. Declining creates no output. `--yes` is an explicit automation bypass.

The policy is harm-focused refusals, helpful answers to legitimate sensitive questions, and honest uncertainty. Supply JSONL conversations tagged `refusal`, `benign`, and `uncertainty`; all three categories are required. Review the labels and answers yourself: structural validation cannot determine whether a dataset is truthful or well-balanced. All examples must fit the context and contain assistant supervision; none are silently dropped. The small `examples/alignment/format-example.jsonl` demonstrates the format, including supplied-context answers and missing-context uncertainty. **It is a format example, not a production alignment dataset or evidence that 14 examples can align a model.**

The output must be a new directory outside the source model folder:

- `original/`: byte-verified copy of the chosen checkpoint, plus its tokenizer/config. No additional refusal training; an already instruction-tuned source may already refuse.
- `aligned/`: a separately trained SFT-compatible model and its tokenizer/config.
- `alignment.json`: policy, checkpoint/data hashes, settings and completion/interruption/failure status.
- `alignment-data.jsonl`: the validated data snapshot actually used for training.

Defaults are one epoch, learning rate 1e-5, batch 1, accumulation 4 and chunked loss. Tune these against a held-out set. Interrupted or diverged runs are not marked complete. Both variants remain usable with `geocentric try --model_dir ...` and further SFT. Training does not guarantee that the original never refuses, that the aligned model always refuses harmful requests, or that either model stops hallucinating.

## Check facts and behavior before claiming improvement

```bash
geocentric check-behavior --model_dir runs/my-alignment/original \
  --compare_dir runs/my-alignment/aligned \
  --data_path examples/alignment/heldout-example.jsonl \
  --output runs/my-alignment/behavior-report.json
```

Use a substantially larger, independent evaluation set for release decisions. Each JSONL case has `category` and `prompt`; factual cases additionally have an `answers` list of accepted short answers. The report records actual greedy responses, checkpoint hashes, context-limited generation budgets, and normalized exact-match scores for factual cases. Refusal, benign-helpfulness and uncertainty cases are deliberately left for human review rather than awarded success based on phrases such as “I cannot.” Exact-match can reject correct paraphrases; it is not a universal factuality metric.

Exact normalized prompts found in the pair's alignment manifest are rejected as training/evaluation overlap. This does not detect paraphrases or pretraining contamination. Long prompts fail explicitly instead of silently losing context. Existing reports are not overwritten.

No training technique can guarantee zero hallucinations on arbitrary prompts. This pass supplies uncertainty supervision, context-grounded examples, independent responses for review, and per-source loss visibility. It does not establish a measured factuality improvement in your partially pretrained model. Production factual answers still need trustworthy evidence, appropriate abstention, and evaluation on the actual use case.

## SFT desktop-memory fix

An 82%-pretrained checkpoint can be instruction-tuned; finishing pretraining is not a
memory-safety requirement. Quality may be limited, but that does not explain a
system freeze. The reported RTX 2060 freeze was not reproduced locally. The code
contained several independent memory hazards:

- SFT defaulted to batch 8 with dense vocabulary loss, unlike the memory-conscious pretraining path.
- Dataset preparation held all rendered conversations, tokenizer encodings and tensors in RAM simultaneously; JSON arrays were read wholesale.
- SFT discarded the compact optimizer choice and always allocated full AdamW state.

SFT now defaults to batch 1, loss chunks of 256, zero loader workers and compilation
off. CUDA enables activation checkpointing and caps PyTorch allocations at 75% of
GPU memory. A static weights/gradients/optimizer-state estimate rejects obviously
oversized configurations before moving the model to CUDA. Activations and external
CUDA allocations can still cause OOM; this is headroom, not a guarantee against
NVIDIA driver or desktop failures. Run the CUDA validation on the actual machine.

`--optimizer auto` retains saved compact EPICYCLE settings, or uses AdamW when none
exist. `--optimizer capacity` explicitly requests factored RingAdamW; `balanced`
is also available. These approximate optimizers change update behavior relative to
AdamW. SFT starts fresh optimizer moments, as before; it does not resume pretraining
momentum. The pipeline uses a separate batch-1, uncompiled SFT phase.

Tokenization now streams JSON/JSONL one conversation at a time and writes int32
examples to scratch storage under the output directory's `sft-cache/`, rather than
holding the whole tokenized dataset in RAM or relying on a potentially RAM-backed
`/tmp`. Only offsets remain resident. Scratch files are removed on normal cleanup;
a forced kill may leave scratch directories that can be removed after the process
has stopped. Oversized individual records are rejected. Whole conversation BPE,
assistant masking and overlong-conversation behavior are retained.

Checkpoint loading uses memory mapping on CPU first, so unused pretraining optimizer
storage need not be read into RAM. GPU transfer occurs after preparation. A status
file is created before loading/tokenizing, with explicit startup messages. Periodic
SFT checkpoints are saved every 100 updates (`--save_every`), in addition to epoch
checkpoints. An OOM during training records failure without attempting another large
GPU operation or labeling partially updated weights as a completed model. A new
output directory receives its tokenizer, making it usable for chat.

Before trying again, stop pretraining and any inference server on the same GPU.
For an initial conservative attempt, add these to your SFT command:

```bash
--batch_size 1 --gradient_accumulation_steps 4 --loss_chunk_size 64 \
--gradient_checkpointing --compile off --num_workers 0 --save_every 25
```

Use a separate `--output_dir` to keep the pretraining run's metrics separate.
This fix passed local regression tests including streaming parsing, disk-backed
supervision, startup status and compact-optimizer SFT. It has not yet been exercised
on the reported RTX 2060 desktop. No throughput improvement is claimed; these
changes prioritize fitting in memory.
