# Buffer and data pipeline refinement

This pass removes repeated allocations and file operations. It changes neither
the model architecture nor its training objective, checkpoint format, optimizer,
resume step or training schedule. The RTX 2060 memory cap, activation checkpointing
and one-example SFT microbatch default remain in place.

## Changes

- SFT caches use a lazy read-only memory mapping instead of opening, seeking,
  reading and closing the file for each sample. File-backed pages are managed by
  the OS; the whole dataset is not copied into a Python array. Returned tensors
  own their converted int64 data, so edits cannot corrupt the cache. Spawned
  workers reopen the mapping instead of serializing the mapped dataset; forked
  workers detect their process ID and reopen it too.
- One-example collation returns batch-shaped views without padding allocations.
  Larger batches allocate the two output tensors once and copy each example into
  its row. Padding and ignored-label values are unchanged.
- SFT validation selects supervised positions and counts tokens on CPU, as training
  already does. A single loss readback handles both accumulation and finite checks.
- Batch generation allocates history once and caps KV capacity at the prompt plus
  requested output, bounded by context. Previously it concatenated history on each
  token and allocated full-context KV buffers even for short outputs. Streaming
  generation already had bounded buffers and is unchanged.

## Measurements and limits

On the local M4 with PyTorch 2.8, the CPU cache-loading/collation microbenchmark
increased from 43,287 to 161,181 examples/s: **3.72×**, or **272% higher throughput**.
This isolates a small CPU component and is not an end-to-end SFT gain.

Synthetic MPS greedy batch generation increased from 369 to 558 output tokens/s:
**51.1% higher throughput**. Outputs matched exactly. The test includes prefill,
uses 64 prompt + 64 output tokens, 4 layers, width 256, 16,000 vocabulary entries
and 1,024 context. It alternates execution order and excludes a warm-up repetition.
KV buffer capacity in that test drops from 1,024 to 128 positions, **87.5% less KV
storage**, not 87.5% less total model memory. Benefits depend on prompt/output length.

Raw samples are under `research/benchmarks/{buffered-generation,mapped-sft-loading}-m4.json`.
CUDA/RTX 2060 gains have not been measured. These numbers cannot be added to the
previous SFT compute speedup and do not imply improved model quality.

Run short synthetic benchmarks without touching trained checkpoints:

```bash
PYTHONPATH=. .venv/bin/python scripts/data_speed_bench.py
PYTHONPATH=. .venv/bin/python scripts/generation_speed_bench.py --device cuda
```

Validation covers masked padding, independent cache tensors, cache pickling and
spawned-worker loading, exact batched greedy output versus the previous algorithm,
zero output requests and context limits. The full suite passed 276 tests with 14
hardware/optional skips. No long training run was started.
