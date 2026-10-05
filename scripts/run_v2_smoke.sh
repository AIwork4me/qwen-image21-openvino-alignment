#!/usr/bin/env bash
# Tier 1: 1-3 prompt smoke validation (encoders + one trajectory prompt).
set -euo pipefail
cd "$(dirname "$0")/.."
export V2_VALDATA=${V2_VALDATA:-/valdata}
source /workspace/qwen-image21-v2-venv/bin/activate
python scripts/v2_check_environment.py
python scripts/v2_freeze_latents.py
python scripts/v2_run_encoder_arm.py --arm R_FP32 --suite smoke
python scripts/v2_run_encoder_arm.py --arm R_BF16 --suite smoke
python scripts/v2_run_encoder_arm.py --arm C_BF16 --suite smoke
python scripts/v2_run_encoder_arm.py --arm C_INT8 --suite smoke
python scripts/v2_run_encoder_arm.py --arm C_W4A8 --suite smoke
python scripts/v2_compare.py --suite smoke
python scripts/v2_trajectory.py --suite smoke --pids S01 --steps 25
echo "TIER1 SMOKE COMPLETE"
