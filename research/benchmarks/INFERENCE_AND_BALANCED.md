# LAN inference and balanced momentum pass

Baseline: Git revision `3771334`. Implementation leaves model parameter shapes and
existing EPICYCLE preset behavior unchanged. Measurements below are on an Apple M4,
PyTorch 2.8.0/MPS BF16; **no RTX 2060 measurements were available for this pass**.

## What changed

- `geocentric try` serves a bundled ChatGPT-style interface on port 8000, with
  localhost and LAN/Wi-Fi URLs. Original Geocentric logo and mascot are bundled.
  Streaming, cancellation, regeneration, edited turns, searchable local history,
  export, themes and sampling settings work without Node or a cloud service.
  This is a local text chat application, not feature parity with ChatGPT's service.
- Bounded, reusable KV storage removes repeated concatenation of attention keys
  and values. Cache contents remain private to each generation. A gradient-enabled
  fallback retains differentiability. Allocation reuse does not make attention
  constant-time: each token still attends over its available context.
- Shared candidate-space sampling applies top-p/min-p to the selected top-k
  candidates rather than sorting the full vocabulary. Exact-k cutoff ties and
  random-number mapping can change sampled output; greedy equivalence was checked.
- Streaming reserves context space, buffers incomplete Unicode characters,
  supports cooperative stop, and reports timing, truncation and finish reason.
- Experimental `--epicycle balanced` partitions every tensor's first moment into
  rotating slices, including dominant embeddings. It retains factored second
  moments and update clipping. This is an optimizer extension, not a claim of
  research novelty or preserved convergence. Existing optimizers remain default.
- An explicit missing checkpoint now raises an error instead of quietly selecting
  another checkpoint. Terminal chat honors checkpoint and dtype options too.
- Preserved the pre-existing local tokenizer self-copy fix and macOS training
  watcher support while updating from GitHub.

## Inference result

[Raw measurements](inference-m4.json): identical random initialization, 32k
vocabulary, 6 layers, width 384, 64-token prompt and 64 generated tokens. Top-k 50,
top-p .95, min-p .05, repetition penalty 1.25. One warmup pair, five measured pairs,
alternating arm order, accelerator synchronization. Background trainers were
stopped before this measurement.

| Implementation | Median tokens/sec |
| --- | ---: |
| Previous revision | 74.26 |
| This pass | 88.27 |

**18.87% higher inference throughput**, or about **15.87% less elapsed time**.
Greedy outputs matched for the 16-token equivalence check. Random-weight timing is
not evidence of improved intelligence or language quality. This benchmark measures
model generation, not HTTP overhead or full user-perceived latency.

## Training memory and speed

[Structural state measurement](balanced-m4.json), 11,208,960 parameters, four rings:

| Optimizer | Maximum resident optimizer state |
| --- | ---: |
| Existing capacity | 32,977,408 bytes |
| New balanced | 11,418,368 bytes |

**65.38% less optimizer state** in this embedding-heavy configuration. Actual model
capacity also depends on weights, gradients, activations and temporary update
buffers; this is not a 65% total-VRAM reduction or a measured larger-model claim.
The saved timing run overlapped two background trainers and is explicitly marked
invalid for speed comparisons. It suggested overhead, so balanced stays opt-in.
After the user requested training stop, no additional training runs were launched.
There is **no defensible training-speed percentage for this pass yet**. Run the
systems benchmark on the RTX 2060 before choosing balanced for a long experiment.

The previous pass's controlled M4 fold benchmark was about 10% faster; its separate
short real-text comparison was about 15.6% faster effective throughput against its
own baseline (see [previous report](REPORT.md)). These are different experiments.
The user's reported RTX increase from 16–17k to a usual 20k tokens/sec corresponds
to **17.6–25%**, approximately **21.2%** versus the old midpoint of 16.5k. That is a
user observation, not a controlled benchmark, and must not be added to the 18.9%
inference gain.

## Repetition investigation

[Local checkpoint probe](repetition-local.json): `geocentric_m4_16m_pretrained_best.pt`,
step 400, prompt “The purpose of learning is”. Greedy decoding without a repetition
penalty entered a sustained “the same as” cycle. Two seeded sampled comparisons
(temperature .8, penalties 1 and 1.25) did not show that sustained loop. One stopped
naturally after 21 tokens; the other reached the 64-token limit. This small probe
shows susceptibility to decoding loops, not a general quality improvement.

Cached and full-prefix inference selected the same top token on this prompt;
BF16 maximum logit difference was .0625. Separate FP32 regression tests check
cached versus full logits over multiple chunks and batches. Production packed
training targets are shifted by one token, not identical to their inputs. Existing
loss/gradient, dataset, and checkpoint tests continue to cover the training path.

There is no exact sample, checkpoint, loss curve or remote access for the user's
53%-complete RTX run, so its root cause remains unconfirmed. Completion percentage
alone is not a diagnosis. Check held-out loss trends, corpus duplication and
whether a base model is being given an instruction-tuned chat template. The UI
selects continuation mode for base checkpoints. Its six-cycle guard visibly stops
sustained 1–4 token repetition; it does not prevent every repeated word or repair
training. Use `scripts/repetition_diagnostic.py` against the affected checkpoint.

## Compatibility and verification

Old parameter names/shapes and default optimizer paths remain compatible. Resume
existing training with the same optimizer configuration. Do not switch an existing
optimizer to balanced and expect identical continuation; mismatched metadata is
rejected. Tests cover exact resume through rotations, legacy metadata without the
new field, single-ring equivalence, bounded cache reuse, stream cancellation,
Unicode, context limits, malformed HTTP requests, busy handling and local assets.
CUDA FP16 integration cases include the balanced optimizer, with explicit skips
when hardware is absent. The RTX 2060 retains the existing FP16 auto-selection.

The wheel includes HTML, JS, CSS and both artwork files. JavaScript syntax and
actual HTTP/NDJSON generation were checked. Automated visual/browser interaction
QA was not run. Linux CI results are recorded in the delivery message; GPU tests
must run on NVIDIA hardware before claiming RTX/Ubuntu GPU validation.
