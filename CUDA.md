# Ubuntu 26.04 and RTX 2060

The RTX 2060 is a Turing GPU with compute capability 7.5
([NVIDIA hardware list](https://developer.nvidia.com/cuda-gpus)). Training keeps FP32
master weights and automatically selects **FP16 autocast with gradient scaling** on
Turing. BF16 compute is reserved for hardware that supports it natively. ARMILLARY's
BF16 state storage is converted to FP32 for optimizer arithmetic.

Attention uses PyTorch SDPA and allows its supported backend to be selected. Eager
training needs no FlashAttention-2, BF16 tensor cores, custom CUDA extensions, or
CUDA compiler. `torch.compile` is optional; start with `--compile off` when validating
a new wheel/driver combination.

## Install

Use an NVIDIA driver compatible with your chosen PyTorch CUDA runtime. Confirm
`nvidia-smi` works. Ubuntu's system Python is managed by the OS; use a virtual
environment for project dependencies.

```bash
sudo apt update
sudo apt install python3-venv git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled torch wheel using the Linux/Pip command from the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e '.[dev]'
bash scripts/validate_cuda.sh
```

The validator executes a CUDA matrix multiplication and prints OS, Python, PyTorch,
CUDA runtime, GPU, compute capability, and wheel architectures. It then runs the
complete test suite, including FP16 gradient comparisons, growth, folding,
checkpointing, fused AdamW, ARMILLARY, batch probing, and compilation. It fails when
CUDA is unavailable instead of presenting skipped tests as a GPU pass. Finally it
writes a synchronized benchmark to `runs/cuda-validation/systems.json`.

If your Python has no compatible CUDA wheel, use a supported Python in a separate
environment following the selector's compatibility list. The toolkit number from
`nvidia-smi` describes driver capability; the installed torch runtime is printed
separately by the validator.

## First run

For the common 6 GB RTX 2060 configuration, begin with a bounded run and measured
batch sizing. The GPU name is not used to assume available VRAM.

```bash
geocentric pretrain --data_path corpus.jsonl --preset 120m \
  --epicycle memory --dtype auto --compile off --batch_size 0 \
  --gradient_accumulation_steps 8 --max_steps 100 --no_watermark \
  --output_dir runs/rtx2060-smoke
```

Batch probing uses the selected optimizer and loss at full depth/context, measures
against free memory, and restores weights and CPU/CUDA RNG. Add
`--gradient_checkpointing` when activations still limit capacity. `--epicycle capacity`
also factors second moments; it is experimental and changes optimization.
`--epicycle speed --loss_chunk_size 0` prioritizes throughput when memory permits.

Snapshots cost host RAM. `--snapshot_every 0` avoids periodic CPU weight copies and
uses on-disk recovery. Saved checkpoints preserve optimizer rotation and token counts.
Legacy checkpoints cannot recover exact historical token counts; resumed data order
is not a bitwise replay of an uninterrupted run.

## Validation status

Local CPU/MPS results are in [the report](research/benchmarks/REPORT.md). GitHub CI
has a native Ubuntu 26.04 container job using packaged Python and a separate
Ubuntu/Python 3.11 job. These are CPU tests, not NVIDIA driver validation.

A CUDA job supports manual dispatch on a registered runner labelled
`self-hosted, linux, x64, gpu`. Actual RTX 2060 performance and compatibility remain
unverified until the validator passes on that machine. M4 timings must not be
extrapolated to Turing.
