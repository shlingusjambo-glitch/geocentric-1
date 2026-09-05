#!/usr/bin/env bash
# Invoke from the repository root after activating its Python environment.
set -euo pipefail
python - <<'PY'
import json, platform, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build and a compatible NVIDIA driver.")
device = torch.device("cuda")
# Execute a kernel: is_available alone does not prove the wheel supports this GPU.
x = torch.randn(64, 64, device=device)
(x @ x).sum().item()
torch.cuda.synchronize()
print(json.dumps(dict(os=platform.platform(), python=platform.python_version(),
    torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
    wheel_architectures=torch.cuda.get_arch_list()), indent=2))
PY
python -m pytest -q
python scripts/training_systems_bench.py --device cuda --dtype auto \
  --output runs/cuda-validation/systems.json "$@"
