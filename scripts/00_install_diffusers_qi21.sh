#!/usr/bin/env bash
# Vendor Qwen-Image-2.1 support into installed diffusers 0.40.0 (idempotent).
#
# v2 note (fresh-machine rerun): jsdelivr CDN is unreachable through the lab
# egress proxy; files are fetched through the GitHub REST API (gh api) instead.
# The vendored classes must be registered in diffusers' lazy _import_structure
# (models/__init__.py); eager imports cause circular-import failures.
set -e
REF=8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd
DST=$(python -c "import diffusers, os; print(os.path.dirname(diffusers.__file__))")
mkdir -p $DST/pipelines/qwenimage21
for f in pipelines/qwenimage21/pipeline_qwenimage21.py pipelines/qwenimage21/__init__.py \
         models/autoencoders/autoencoder_kl_qwenimage21.py models/transformers/transformer_qwenimage21.py; do
  for i in 1 2 3 4 5; do
    gh api "repos/huggingface/diffusers/contents/src/diffusers/$f?ref=$REF" --jq '.content' 2>/dev/null \
      | base64 -d > "$DST/$f" && [ -s "$DST/$f" ] && break
    sleep 3
  done
  [ -s "$DST/$f" ] || { echo "FAILED to fetch $f"; exit 1; }
done
# lazy registration in models/__init__.py (idempotent)
grep -q 'autoencoder_kl_qwenimage21' $DST/models/__init__.py || \
  sed -i '/_import_structure\["autoencoders.autoencoder_kl_qwenimage"\] = \["AutoencoderKLQwenImage"\]/a\    _import_structure["autoencoders.autoencoder_kl_qwenimage21"] = ["AutoencoderKLQwenImage21"]' $DST/models/__init__.py
grep -q 'transformer_qwenimage21' $DST/models/__init__.py || \
  sed -i '/_import_structure\["transformers.transformer_qwenimage"\] = \["QwenImageTransformer2DModel"\]/a\    _import_structure["transformers.transformer_qwenimage21"] = ["QwenImage21Transformer2DModel"]' $DST/models/__init__.py
python - <<'PY'
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
from diffusers.models import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
print("vendored diffusers Qwen-Image-2.1 support OK")
PY
