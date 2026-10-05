#!/usr/bin/env bash
# Tier 5: everything incl. 2K production, prompt-length regression, perf, reports.
set -euo pipefail
cd "$(dirname "$0")/.."
export V2_VALDATA=${V2_VALDATA:-/valdata}
source /workspace/qwen-image21-v2-venv/bin/activate
bash scripts/run_v2_core.sh
bash scripts/run_v2_trajectory.sh
bash scripts/run_v2_quality.sh
python scripts/v2_population.py --suite production_30 --arms C_BF16,C_INT8,C_W4A8 --steps 40 --resolution 2048 --track-resources --out-tag _2k
python scripts/v2_semantic_metrics.py --group population --suite production_30 --pattern "P*"
python scripts/v2_ocr_metrics.py --group population --suite production_30 --pattern "P*"
python scripts/v2_prompt_length.py
python scripts/v2_perf.py
python scripts/v2_verify_artifacts.py
python scripts/v2_build_reports.py
echo "TIER5 FULL COMPLETE"
