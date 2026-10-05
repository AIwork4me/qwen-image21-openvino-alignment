#!/usr/bin/env bash
# v2 environment bootstrap: ROCm 7.14.0 baseline + mandated gfx1100 PyTorch build.
# Tested on Ubuntu 24.04 (AMD EPYC + Radeon PRO W7900D / gfx1100).
set -euo pipefail

VENV=${VENV:-/workspace/qwen-image21-v2-venv}
CONSTRAINTS=${CONSTRAINTS:-/workspace/torch-rocm-constraints.txt}

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

echo "== [3/6] ROCm SDK/runtime 7.14.0 (pip, gfx1100) =="
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm[libraries,devel,device-gfx1100]==7.14.0"

echo "== [4/6] mandated PyTorch build (torch 2.12.0+rocm7.14.1 / torchvision 0.27.0+rocm7.14.1) =="
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" \
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1"

echo "== [5/6] restore 7.14.0 runtime baseline =="
# The torch wheels declare rocm==7.14.1 deps; the v2 mandate is a 7.14.0
# SDK/runtime baseline, so we pin the runtime back (expected pip metadata
# conflict with torch is documented in artifacts/v2/environment/).
cat > "$CONSTRAINTS" <<'EOF'
torch==2.12.0+rocm7.14.1
torchvision==0.27.0+rocm7.14.1
rocm==7.14.0
rocm-sdk-core==7.14.0
rocm-sdk-libraries==7.14.0
rocm-sdk-device-gfx1100==7.14.0
rocm-sdk-devel==7.14.0
EOF
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  -c "$CONSTRAINTS" \
  "rocm[libraries,devel,device-gfx1100]==7.14.0" || true

echo "== [6/6] hard gate =="
python "$(dirname "$0")/v2_check_rocm_environment.py"
