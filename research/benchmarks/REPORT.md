# Training efficiency validation — 2026-09-05

Hardware: Apple M4, 8 GPU cores, 16 GB unified memory. Python 3.9.6, PyTorch 2.8.0,
MPS BF16 autocast and FP32 parameters. No NVIDIA GPU was available locally.

## Systems ablation

11,208,960 parameters; 4 layers, width 256, 32,000 vocabulary; batch 2, context 512.
Three warmup updates excluded, twelve measured updates per arm. Same initialization
and random next-token pairs. Each arm processes **12,288 measured targets**.

| arm | tokens/second | maximum sampled forward allocation (MiB) | maximum persistent optimizer state (MiB) |
|---|---:|---:|---:|
| dense | 9,857 | 497.3 | 85.5 |
| chunked | 8,258 | 355.4 | 85.5 |
| fold_dense | 10,844 | 423.7 | 85.5 |
| fold | 8,964 | 284.1 | 85.5 |
| capacity | 8,597 | 249.5 | 31.4 |

Folding alone (`fold_dense`) achieved about **1.10x throughput**. Chunking alone used
about **29% less sampled forward allocation**, at about **16% lower throughput**.
The combined `capacity` arm used about **50% less sampled forward allocation** than
dense AdamW. `fold` uses chunked loss as well as folding.

These are MPS allocation snapshots after forward, not allocator peaks, whole-system
RAM, or capacity guarantees. Intermediate recomputation/optimizer allocations can be
higher. CUDA's harness additionally reports real allocator peaks. No loss values
from this random-token benchmark support intelligence or convergence claims. Arm
order and thermals can affect timings; warmup is excluded but this is a short run.

The tied embedding dominates this small model: four rings are not balanced enough to
justify assuming exactly one-quarter of parameters holds momentum at peak. Persistent
state measurements, including the largest observed ring, are reported instead.

Raw output: [M4 systems JSON](m4-training-systems.json).

```bash
python scripts/training_systems_bench.py --output runs/systems/results.json
```

## Real-text learning smoke experiment

Public-domain *Alice's Adventures in Wonderland*, from
[Project Gutenberg](https://www.gutenberg.org/ebooks/11). Gutenberg header/footer
removed, remaining text partitioned into 4,096-character records; 37 records total.
Data preparation partitions records into train/validation before packing. Adjacent
records belong to the same book: this is not an independent-domain generalization
test. Provenance and source SHA-256: [source metadata](corpus-source.json).

Six layers, width 256, four query heads, two KV heads, vocabulary target 4,000,
context 512, batch two, one microbatch/update, 80 updates. Identical initialization
seed across arms within each trial. Seeds 2026, 2027, 2028; orders baseline/speed/
capacity, capacity/speed/baseline, speed/baseline/capacity. One shared tokenizer and
prepared corpus per trial, outside timed runs. Evaluation at full depth/context
on the held-out records every 20 steps; loss guard disabled. Final partial batches
are retained, explaining the count below the nominal 81,920 tokens.

| arm | mean total seconds | mean final held-out CE | held-out CE range across seeds | targets per trial |
|---|---:|---:|---:|---:|
| baseline | 6.90 | 5.1018 | 5.0939–5.1105 | 80,896 |
| speed | 5.97 | 5.0688 | 5.0674–5.0710 | 80,896 |
| capacity | 6.23 | 5.0261 | 5.0106–5.0360 | 80,896 |

This establishes that all arms train and evaluate on real text; it does not establish
better intelligence. The corpus is tiny, repeated across training, and the run is
far below a useful pretraining budget. Small validation differences are not a
statistical quality claim. Total times include setup, evaluation and checkpoint
writes; initial process/kernel overhead and trial order are material at this scale.
The standalone systems benchmark is a cleaner steady-state timing measurement.

Raw outputs: [seed 2026](realtext-seed2026.json), [seed 2027](realtext-seed2027.json),
[seed 2028](realtext-seed2028.json). Checkpoints and source text are not
included in the commit. The original public-domain source remains downloadable.

```bash
python scripts/epicycle_bench.py --data corpus.jsonl --steps 80   --n_layer 6 --n_embd 256 --n_head 4 --n_kv_head 2 --vocab_size 4000   --block_size 512 --batch_size 2 --accum 1 --eval_every 20   --val_fraction 0.15 --seed 2026 --arms baseline speed capacity   --output runs/realtext
```

## Correctness and platform validation

The local suite covers dense/chunked loss and every parameter gradient, tied weights,
masked targets, EQUANT across chunks, MPS BF16 backward, retained vocabulary
activations, loss normalization, token folding, non-power-of-two contexts, identity
growth, checkpoint round trips, rotating/factored optimizer continuity, no-gradient
ring eviction, cached multi-token attention, and real tiny pretrain/SFT/vision runs.
SFT and vision tests verify an update occurs even when fewer microbatches exist than
the accumulation target. The vision dataset is verified not to be overwritten.

CUDA-specific tests are skipped locally and must pass on NVIDIA hardware before
claiming RTX 2060 validation. Ubuntu 26.04 CPU CI and CUDA validation instructions:
[CUDA guide](../../CUDA.md). A Mac test pass is not an Ubuntu/CUDA test pass.

### Executed Linux CI

[Training implementation CI run](https://github.com/shlingusjambo-glitch/geocentric-1/actions/runs/33999806028)
passed on both Linux jobs. Ubuntu 26.04 container: Python 3.14.4, PyTorch 2.14.0+cpu,
**193 passed, 10 skipped** at training implementation commit `6c66f3e`. The skips are
nine CUDA tests plus the MPS-only test. The NVIDIA job was explicitly skipped.
The local CPU/MPS suite also passes; local AOT-eager compilation successfully ran
chunked forward/backward at two sequence lengths. AOT-eager does not validate CUDA
Inductor kernels. A later release-packaging regression test protects vision datasets.
