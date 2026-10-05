#!/usr/bin/env bash
# v2 environment bootstrap: ROCm 7.14.1 (pip) + mandated gfx1100 PyTorch build.
# Tested on Ubuntu 24.04 (AMD EPYC + Radeon PRO W7900D / gfx1100).
# Reproduces the authoritative v2 stack; terminates with the hard environment gate.
set -euo pipefail

VENV=${VENV:-/workspace/qwen-image21-v2-venv}
CONSTRAINTS=${CONSTRAINTS:-/workspace/rocm7141-torch-constraints.txt}

echo "== [1/6] base OS packages =="
apt-get update -qq || true
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git git-lfs curl wget ca-certificates build-essential cmake ninja-build pkg-config \
    python3 python3-dev python3-pip python3-venv python3.12 python3.12-dev python3.12-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg jq unzip zip zstd htop pciutils numactl

echo "== [2/6] python venv =="
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "== [3/6] ROCm SDK/runtime 7.14.1 (pip, gfx1100) =="
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm[libraries,devel,device-gfx1100]==7.14.1"

echo "== [4/6] mandated PyTorch build (torch 2.12.0+rocm7.14.1 / torchvision 0.27.0+rocm7.14.1) =="
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" \
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1"

echo "== [5/6] lock the torch stack =="
cat > "$CONSTRAINTS" <<'EOF'
torch==2.12.0+rocm7.14.1
torchvision==0.27.0+rocm7.14.1
EOF
python - <<'PY'
import torch, torchvision
assert torch.__version__.startswith("2.12.0+rocm7.14.1"), torch.__version__
assert torchvision.__version__.startswith("0.27.0+rocm7.14.1"), torchvision.__version__
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| HIP", torch.version.hip)
PY

echo "== [6/6] hard gate =="
python "$(dirname "$0")/v2_check_environment.py"
