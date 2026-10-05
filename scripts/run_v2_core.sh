#!/usr/bin/env bash
# Tier 2: canary_10 numerical/layer validation across all arms + OpenVINO.
set -euo pipefail
cd "$(dirname "$0")/.."
export V2_VALDATA=${V2_VALDATA:-/valdata}
source /workspace/qwen-image21-v2-venv/bin/activate
python scripts/v2_check_environment.py
[ -f /valdata/models/v2_ov_fp32/language_model_fp32.xml ] || python scripts/v2_export_ov.py
for arm in R_FP32 R_BF16 C_BF16 C_INT8 C_W4A8; do
  python scripts/v2_run_encoder_arm.py --arm $arm --suite canary
done
python scripts/v2_run_encoder_arm.py --arm C_BF16 --suite canary --repeats 3
python scripts/v2_run_encoder_arm.py --arm R_BF16 --suite canary --repeats 3
python scripts/v2_export_ov_exact.py --arm O_INT8_EXACT
python scripts/v2_export_ov_exact.py --arm O_W4A8_EXACT
python scripts/v2_run_ov_arm.py --arm O_FP32 --suite canary
python scripts/v2_run_ov_arm.py --arm O_BF16 --suite canary
python scripts/v2_run_ov_arm.py --arm O_INT8_EXACT --suite canary
python scripts/v2_run_ov_arm.py --arm O_W4A8_EXACT --suite canary
python scripts/v2_weight_parity.py
python scripts/v2_compare.py --suite canary
echo "TIER2 CORE COMPLETE"
