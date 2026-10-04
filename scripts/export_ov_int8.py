#!/usr/bin/env python3
"""Reproduce the customer-style INT8 weight compression (arm O3r).

The customer artifact (OpenVINO/Qwen3-VL-8B-Instruct-int8-ov style) applies:
  - INT8_ASYM weight compression on linear layers (default quant, ratio 1.0)
  - runtime dynamic activation quantization (plugin-driven, group 32)
  - INT8-compressed token-embedding table (artifact bin = 622,633,732 B ~= 151936*4096 int8)
On this host that artifact's IR fails for prompts >40 tokens (eltwise shape bug in
OV 2026.4.x), so we reproduce the recipe on the clean wrapper export (which works):
  - nncf.compress_weights(INT8_ASYM, ratio=1.0, group_size=-1) on language_model_fp32
  - row-wise int8-asym quantized embedding table (lookup + dequant in numpy)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json

import openvino as ov
import nncf


def quantize_embed_table_int8(table_fp32):
    """Row-wise asymmetric int8 quantization of [vocab, dim] embedding table."""
    mn = table_fp32.min(axis=1, keepdims=True)
    mx = table_fp32.max(axis=1, keepdims=True)
    scale = (mx - mn) / 255.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round((table_fp32 - mn) / scale), 0, 255).astype(np.uint8)
    return q, mn.astype(np.float32), scale.astype(np.float32)


def main():
    cfg = load_cfg()
    src = cfg["model"].get("ov_clean_export_dir") or "/valdata/models/ov_clean_fp32"
    out = "/valdata/models/ov_clean_int8"
    os.makedirs(out, exist_ok=True)

    core = ov.Core()
    print("reading", src)
    lm = core.read_model(os.path.join(src, "language_model_fp32.xml"))
    t0 = time.time()
    compressed = nncf.compress_weights(
        lm,
        mode=nncf.CompressWeightsMode.INT8_ASYM,
        ratio=1.0,
        group_size=-1,
    )
    print(f"compressed in {time.time()-t0:.0f}s")
    ov.save_model(compressed, os.path.join(out, "language_model_int8.xml"), compress_to_fp16=False)
    sz = os.path.getsize(os.path.join(out, "language_model_int8.bin")) / 2 ** 30
    print(f"saved language_model_int8 ({sz:.2f} GiB)")

    # embedding table int8 (mirrors artifact's 622MB embed blob)
    from safetensors.torch import safe_open
    import glob, torch
    key = "model.language_model.embed_tokens.weight"
    table = None
    for f in sorted(glob.glob(os.path.join(cfg["model"]["snapshot_path"], "text_encoder", "model-*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            if key in sf.keys():
                table = sf.get_tensor(key).float().numpy()
                break
    q, zp, scale = quantize_embed_table_int8(table)
    np.savez(os.path.join(out, "embed_table_int8.npz"), q=q, zero_point=zp, scale=scale)
    print("saved embed_table_int8.npz", q.shape, q.dtype)

    save_json({"source": src, "tool": f"nncf {nncf.__version__} compress_weights",
               "mode": "INT8_ASYM", "ratio": 1.0, "group_size": -1,
               "embed": "row-wise int8 asym + numpy dequant",
               "note": "reproduction of customer quantization recipe on working clean export"},
              os.path.join(out, "conversion_manifest.json"))
    print("DONE")


if __name__ == "__main__":
    main()
