# EPICYCLE: local pretraining efficiency

EPICYCLE combines training methods along different resource axes. These are
engineering contributions to this stack, not a claim that efficient pretraining
began here. A faster step does not establish better language modeling, and a smaller
optimizer does not establish unchanged convergence.

| preset | depth growth | context folding | selective loss | optimizer | loss head |
|---|---|---|---|---|---|
| off | no | no | no | AdamW | dense |
| speed | yes | yes | no | AdamW | dense |
| quality | no | yes | yes | AdamW | dense |
| memory | yes | yes | no | rotating momentum | chunked |
| full | yes | yes | yes | rotating momentum | chunked |
| capacity | yes | yes | no | factored rotating momentum | chunked |

Override the loss head with `--loss_chunk_size`: zero is dense; a positive value
limits tokens per projection. The architecture and weight names do not change.
Inference still returns logits through the original API.

## DEFERENT: grow without erasing learning

Start with half the layers and activate the rest by 60% of optimizer steps. Initial
active layers retain their scaled random initialization. Sleeping layers have zero
residual output projections, making them identities even during full-depth evaluation.
They are skipped during training and get no gradients or weight decay. A newly active
block learns its output projection first, followed by its internal projections.

Older code zeroed all active output projections when initializing the scheduler,
including trained layers on resume. The scheduler now preserves those weights and
stores the highest activated depth in checkpoints and recovery snapshots. Metadata
is optional; older checkpoints still load. The legacy fallback infers active depth
from the requested schedule: keep the preset and step budget unchanged for old runs.
Late layers still receive fewer updates. Relevant prior work:
[Net2Net](https://arxiv.org/abs/1511.05641) and
[progressive stacking](https://proceedings.mlr.press/v97/gong19a.html).

## HORIZON: retain tokens while shortening attention

The old implementation cropped a window and discarded its suffix. The new default
folds windows into shorter independent sequences along the batch dimension. A
`[2, 1024]` batch becomes `[8, 256]`: all 2,048 input/target pairs remain. A final
partial segment is right-padded with ignored targets. Final context reaches the
configured length even when it is not a power of two.

Folding preserves tokens and next-token labels, **not full-context conditioning**.
Linear layers process the same tokens; attention processes shorter segments. EQUANT
selection sees all folded segments together within each microbatch. Batch sizing
still probes full depth and context with the selected optimizer and loss path.
Non-divisible lengths have padding overhead. `horizon_mode="crop"` in
`EpicycleConfig` reproduces the legacy crop behavior.

## Exact chunked vocabulary loss

Training normally materializes `[batch * context, vocabulary]` logits and associated
cross-entropy intermediates. The new path checkpoints both the tied output projection
and cross entropy. Keeping token-sized loss vectors lets backward recompute one
projection chunk at a time. Chunking without checkpointing would retain every chunk's
backward intermediates and would not solve the memory problem.

The portable PyTorch implementation needs no custom CUDA extensions. Loss computation
remains FP32 under autocast. It supports masked targets, sum/mean/per-token reductions,
tied embedding gradients, and global trimmed loss. A fully masked chunked mean is a
differentiable zero. This uses
[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint).
[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) addresses the same memory problem
with specialized kernels; this implementation does not claim its fusion or speed.

Chunking reduced memory and cost throughput on the measured M4 workload. Speed presets
therefore retain dense loss. SFT and vision can opt in with `--loss_chunk_size 256`.

## EQUANT: experimental trimmed selection

After 15% of training, keep the hardest 65% of valid token losses below the top 2%.
Reported loss remains the mean over all valid tokens. Chunking preserves the global
per-token vector, so selection is not independently applied per projection chunk.

High loss does not prove corruption: rare facts and difficult reasoning can also have
high loss. Selection may harm calibration or rare-token learning. It saves no dense
transformer FLOPs, and quality benefit remains a hypothesis requiring held-out and
downstream tests. Related work: [Rho-1](https://arxiv.org/abs/2404.07965), which uses
reference-model scoring.

## ARMILLARY: state capacity and persistence

Parameters are assigned greedily to rings by size. One ring holds FP32 first moments;
all parameters with gradients update every step, with momentum-free adaptive updates
while cold. Second moments are stored in BF16 and updated in FP32.

Checkpoints now preserve global rotation step, ring count, dwell, and factor mode.
Reloading restores intentional storage dtypes; generic optimizer loading would cast
BF16 state to FP32. Cold momentum is freed before new allocation, including tensors
without gradients. Changing optimizer families on resume raises an actionable error.

An ideally balanced four-ring model needs approximately `8 + 2 + 4/4 = 11` bytes per
parameter for weights, gradients, and persistent state, versus 16 for FP32 AdamW.
**Tensors are not split between rings.** A dominant embedding can make the largest
ring much larger than `parameters/rings`. Runtime estimates use actual largest-ring
load, and benchmarks measure state. Activations, scratch space, snapshots, and
allocator overhead are additional costs. `rings=1` needs **14 bytes per parameter
including gradients**. BF16 state is approximate, not a guarantee of identical AdamW
convergence. Default dwell is 200 steps.

### Experimental capacity preset

Matrices use FP32 row and column second moments. Their outer product, divided by the
row mean, reconstructs the scale. Vectors retain dense BF16 second moments. Bias
correction is per parameter. Factored updates are clipped to RMS at most one to bound
unusually large approximate updates.

This combines [Adafactor-style factorization](https://arxiv.org/abs/1804.04235) with
rotating first-moment residency. It is neither full Adafactor nor exact AdamW. A matrix
with `r*c` entries needs `4*(r+c)` second-moment bytes instead of `2*r*c`. Reconstructed
scales still require matrix-sized temporary memory. Evaluate the optimization trade
before committing a long run to this preset.

## Measurement

```bash
python scripts/training_systems_bench.py
python scripts/epicycle_bench.py --data corpus.jsonl --steps 400 \
  --arms baseline speed capacity --output runs/epicycle-comparison
```

The systems test synchronizes accelerator timing, excludes warmup, uses identical
random inputs and initialization, and reports state separately from activations.
Random-token losses are not quality evidence. CUDA reports allocator peak; MPS
reports sampled allocation values only.

The real-text harness prepares one tokenizer/corpus before timing any arm, resets
seeds per arm, disables recovery, and measures full-depth/full-context held-out loss.
Time-to-loss uses held-out observations, not changing training contexts. Existing
output directories require explicit `--overwrite`.
See [measured results](research/benchmarks/REPORT.md).

Historical crop-based M4 claims are superseded: the old harness charged tokenizer
setup to the baseline, did not reset initialization seeds, and compared training-loss
crossing times. Those results cannot establish a reliable speed-to-quality advantage.
Beating larger cloud-trained models remains a research objective, not an outcome
established by these short tests.

## Selected vocabulary replay (opt-in)

`--epicycle selective --compile off` keeps the `quality` preset's policy and enables
selected-row backward replay in chunked vocabulary cross-entropy. Forward scores
all tokens; backward compacts and replays only rows that EQUANT gives nonzero
gradient. It preserves the first-order objective, with different floating-point
accumulation order. Existing presets and checkpoint tensor shapes remain unchanged.
For an existing `quality` or `full` run, add `--equant_sparse_replay --compile off`
to its original command to keep the optimizer choice. Compiled execution falls
back to ordinary chunked replay.

Matched M4 synthetic update benchmarks measured 4.87–5.34% higher throughput.
This is neither a CUDA result nor a measured quality gain. See the
[training refinement guide](docs/TRAINING_REFINEMENT.md) for raw measurements,
limitations, per-source validation, and the separate optional alignment command.
