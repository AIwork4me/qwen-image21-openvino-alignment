#!/usr/bin/env bash
# Tier 4: 100-prompt population + semantic + OCR + VLM judge + ablations.
set -euo pipefail
cd "$(dirname "$0")/.."
export V2_VALDATA=${V2_VALDATA:-/valdata}
source /workspace/qwen-image21-v2-venv/bin/activate
python scripts/v2_check_environment.py
# encoder tensors for the full population (+ OV subset) + seed variants
for arm in C_BF16 C_INT8 C_W4A8; do
  python scripts/v2_run_encoder_arm.py --arm $arm --suite alignment_100
done
# 3-seed subset: same embeddings, three frozen latents (20261001/02/03)
for SEEDTAG in "_s20261002" "_s20261003"; do
  python scripts/v2_population.py --suite canary --arms C_BF16,C_INT8,C_W4A8 --steps 40 --latent-suffix $SEEDTAG --out-tag _seed$SEEDTAG
done
python scripts/v2_population.py --suite alignment_100 --arms C_BF16,C_INT8,C_W4A8 --steps 25 40
python scripts/v2_semantic_metrics.py --group population --suite alignment_100
python scripts/v2_ocr_metrics.py --group population --suite alignment_100
python scripts/v2_vlm_judge.py --group population --suite alignment_100 --limit 40
python scripts/v2_ablations.py --suite canary
echo "TIER4 QUALITY COMPLETE"
