#!/usr/bin/env python3
"""Exact weight-space parity: dequantized C_INT8 / C_W4A8 vs C_BF16 weights.

Uses the exact comfy_kitchen eager reference dequantization (the same math the
HIP kernels implement) for every quantized tensor in both official artifacts,
then compares effective reconstructed weights against the official BF16
artifact.  Emits:
  artifacts/v2/metrics/int8_weight_parity.csv
  artifacts/v2/metrics/w4a8_weight_parity.csv
  artifacts/v2/metrics/w4a8_codebook_provenance.json
  artifacts/v2/metrics/weight_parity_summary.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

from comfy_kitchen.backends.eager.w4a8_int8 import dequantize_w4a8_int8_weight
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.backends.eager import w4a8_int8 as w4a8_eager


def dequant_int8_convrot(q: torch.Tensor, scale: torch.Tensor, group: int, dtype: torch.dtype) -> torch.Tensor:
    params = TensorWiseINT8Layout.Params(scale=scale.float(), orig_dtype=dtype, orig_shape=tuple(q.shape),
                                         is_weight=True, convrot=True, convrot_groupsize=group)
    return TensorWiseINT8Layout.dequantize(q, params)


def main() -> int:
    t0 = time.time()
    bf16_path = os.path.join(C.ENCODER_DIR, "qwen3vl_8b_bf16.safetensors")
    int8_path = os.path.join(C.ENCODER_DIR, "qwen3vl_8b_int8_convrot.safetensors")
    w4a8_path = os.path.join(C.ENCODER_DIR, "qwen3vl_8b_w4a8.safetensors")

    with safe_open(bf16_path, framework="pt", device="cpu") as bf:
        for arm, path, out_name in (("C_INT8", int8_path, "int8_weight_parity.csv"),
                                    ("C_W4A8", w4a8_path, "w4a8_weight_parity.csv")):
            rows = []
            with safe_open(path, framework="pt", device="cpu") as qf:
                keys = [k for k in qf.keys() if k.endswith(".weight") and not k.startswith("model.visual")
                        and ("comfy_quant" not in k)]
                quant_cfgs = {k[:-len(".comfy_quant")]: json.loads(qf.get_tensor(k).numpy().tobytes())
                              for k in qf.keys() if k.endswith(".comfy_quant")}
                for key in sorted(keys):
                    mod = key[:-len(".weight")]
                    if mod not in quant_cfgs:
                        continue  # BF16 tensors (norms) - identical by construction
                    cfg = quant_cfgs[mod]
                    qw = qf.get_tensor(key)
                    ref = bf.get_tensor(key) if key in bf.keys() else None
                    if ref is None:
                        print(f"WARN: {key} missing in BF16 artifact", file=sys.stderr)
                        continue
                    if cfg["format"] == "int8_tensorwise":
                        scale = qf.get_tensor(mod + ".weight_scale")
                        eff = dequant_int8_convrot(qw, scale, cfg.get("convrot_groupsize", 256), torch.bfloat16)
                        fmt = "int8_convrot"
                    elif cfg["format"] == "asym_w4a8_int8":
                        s_rel = qf.get_tensor(mod + ".weight_s_rel")
                        s_ch = qf.get_tensor(mod + ".weight_s_channel")
                        cb = qf.get_tensor(mod + ".weight_codebook")
                        eff = dequantize_w4a8_int8_weight(
                            qw, s_rel, s_ch, codebook=cb, correction=None,
                            group_size=cfg.get("group_size", 16),
                            convrot_groupsize=cfg.get("convrot_groupsize", 256),
                            output_dtype=torch.bfloat16)
                        fmt = "w4a8"
                    else:
                        raise ValueError(f"{arm} {mod}: unknown format {cfg['format']}")
                    m = C.pair_metrics(ref.float(), eff.float())
                    m.update({"arm": arm, "format": fmt, "tensor": mod,
                              "shape": list(qw.shape), "orig_shape": list(ref.shape),
                              "shape_equal": list(qw.shape) != list(ref.shape)})
                    rows.append({k: m[k] for k in ("arm", "format", "tensor", "shape", "orig_shape", "shape_equal",
                                                   "rel_l2", "cosine", "rmse", "max_abs_err", "mean_abs_err",
                                                   "norm_ratio", "nan", "inf")})
                    print(f"{arm} {mod}: rel_l2={m['rel_l2']:.5f} cos={m['cosine']:.6f}", flush=True)
                    del qw, eff, ref
            out = os.path.join(C.REPO, "artifacts", "v2", "metrics", out_name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"-> {out} ({len(rows)} rows)")

        # codebook provenance: all layers vs _FIXED_LUT
        lut = torch.tensor(w4a8_eager._FIXED_LUT, dtype=torch.float32)
        prov = {"fixed_lut": list(w4a8_eager._FIXED_LUT), "layers": {}, "all_bit_exact_to_fixed_lut": True}
        with safe_open(w4a8_path, framework="pt", device="cpu") as qf:
            cb_keys = [k for k in qf.keys() if k.endswith(".weight_codebook")]
            for k in sorted(cb_keys):
                cb = qf.get_tensor(k)
                bit_exact = bool(torch.equal(cb.view(torch.int32), lut.view(torch.int32)))
                prov["layers"][k] = {"bit_exact": bit_exact, "values": [float(v) for v in cb]}
                prov["all_bit_exact_to_fixed_lut"] &= bit_exact
        C.save_json(prov, os.path.join(C.REPO, "artifacts", "v2", "metrics", "w4a8_codebook_provenance.json"))
        print(f"codebook provenance: {len(cb_keys)} layers, all_bit_exact={prov['all_bit_exact_to_fixed_lut']}")

    summary = {"captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "runtime_s": time.time() - t0,
               "experiment_id": "V2-WEIGHT-PARITY"}
    C.save_json(summary, os.path.join(C.REPO, "artifacts", "v2", "metrics", "weight_parity_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
