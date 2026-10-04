#!/usr/bin/env bash
# Vendor Qwen-Image-2.1 support into installed diffusers 0.40.0 (idempotent).
set -e
BASE=https://cdn.jsdelivr.net/gh/huggingface/diffusers@8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd
DST=$(python -c "import diffusers, os; print(os.path.dirname(diffusers.__file__))")
mkdir -p $DST/pipelines/qwenimage21
for f in pipelines/qwenimage21/pipeline_qwenimage21.py pipelines/qwenimage21/__init__.py \
         models/autoencoders/autoencoder_kl_qwenimage21.py models/transformers/transformer_qwenimage21.py; do
  curl -sL --retry 3 -o $DST/$f $BASE/src/diffusers/$f
done
python - <<'PY'
import diffusers
from diffusers import QwenImage21Pipeline
print("QwenImage21Pipeline import OK")
PY
