#!/usr/bin/env python3
"""v2 Section 12: weight-space parity + independent decoder verification.

1. INDEPENDENT W4A8/INT8 decoder implemented from the format spec (no comfy_kitchen):
   verify bit-exact agreement with comfy_kitchen's dequantize on sample layers.
2. Weight-space quantization error vs the official BF16 weights, per layer:
   shape/cosine/rel-L2/RMSE/max-abs/scale parity → int8_weight_parity.csv,
   w4a8_weight_parity.csv.
"""
import csv
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json

BF16 = "/valdata/v2models/text_encoders/qwen3vl_8b_bf16.safetensors"
INT8 = "/valdata/v2models/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
W4A8 = "/valdata/v2models/text_encoders/qwen3vl_8b_w4a8.safetensors"

_FIXED_LUT = (-0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187,
              -0.155210, -0.050720, 0.052541, 0.156985, 0.265284, 0.379533,
              0.502636, 0.638953, 0.794876, 0.980671)


def hdr_of(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n).decode()), 8 + n


def get_tensors(path, names, dt="pt"):
    import safetensors.torch as st
    with st.safe_open(path, framework="pt", device="cpu") as f:
        return {n: f.get_tensor(n) for n in names}


def hadamard256(dtype=torch.float64):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < 256:
        h = torch.kron(h, h4)
    return h / (256 ** 0.5)


def my_int8_convrot_dequant(q, scale, g=256):
    """Spec formula: unrotate(round(q*s) @ H) per group. q [N,K] int8, scale [N,1]."""
    N, K = q.shape
    w = q.float() * scale.float()
    H = hadamard256().to(w.dtype)
    wg = w.reshape(N, K // g, g)
    return torch.matmul(wg, H).reshape(N, K)


def my_w4a8_dequant(qdata, s_rel, s_channel, codebook, group_size=16, g=256):
    """Spec formula: unrotate(round(clamp(LUT[code]*s_rel)) * s_channel)."""
    N, cols = qdata.shape
    K = cols * 2
    packed = qdata.view(torch.uint8).to(torch.int64)
    codes = torch.empty(N, K, dtype=torch.int64)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    vals = codebook[codes]                       # [N,K]
    groups = K // group_size
    s = s_rel.float()
    v = vals.reshape(N, groups, group_size) * s.reshape(N, groups, 1).float()
    v = v.reshape(N, K).round().clamp(-127, 127)  # int8 grid
    v = v * s_channel.float().reshape(N, 1)
    H = hadamard256().to(v.dtype)
    wg = v.reshape(N, K // g, g)
    return torch.matmul(wg, H).reshape(N, K)


def stats(a, b):
    d = (a.double() - b.double())
    na = a.double().reshape(-1)
    return {
        "shape": list(a.shape),
        "cosine": float((na @ b.double().reshape(-1)) / (na.norm() * b.double().norm() + 1e-30)),
        "rel_l2": float(d.norm() / na.norm()),
        "rmse": float((d ** 2).mean() ** 0.5),
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "allclose_1e_5": bool(torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5)),
    }


def main():
    outdir = os.path.join(ROOT, "artifacts", "v2", "metrics")
    os.makedirs(outdir, exist_ok=True)
    H = hadamard256()

    # ---------- independent decoder verification ----------
    verify = {}
    layers_v = ["model.layers.0.self_attn.q_proj", "model.layers.17.mlp.down_proj",
                "model.layers.35.mlp.gate_proj"]
    t = get_tensors(W4A8, [f"{l}.weight" for l in layers_v] +
                    [f"{l}.weight_s_rel" for l in layers_v] +
                    [f"{l}.weight_s_channel" for l in layers_v] +
                    [f"{l}.weight_codebook" for l in layers_v])
    from comfy_kitchen.backends.eager.w4a8_int8 import dequantize_w4a8_int8_weight as ck_w4
    for l in layers_v:
        mine = my_w4a8_dequant(t[f"{l}.weight"], t[f"{l}.weight_s_rel"],
                               t[f"{l}.weight_s_channel"], t[f"{l}.weight_codebook"])
        theirs = ck_w4(t[f"{l}.weight"], t[f"{l}.weight_s_rel"], t[f"{l}.weight_s_channel"],
                       codebook=t[f"{l}.weight_codebook"], group_size=16, convrot_groupsize=256,
                       output_dtype=torch.float32)
        s = stats(mine, theirs)
        s["max_abs_diff_vs_kitchen"] = s.pop("max_abs")
        verify[f"w4a8:{l}"] = s
        print(f"W4A8 decode check {l}: rel_l2={s['rel_l2']:.3e} max={s['max_abs_diff_vs_kitchen']:.3e} allclose={s['allclose_1e_5']}")

    layers_i = ["model.layers.0.self_attn.q_proj", "model.layers.17.mlp.down_proj"]
    t = get_tensors(INT8, [f"{l}.weight" for l in layers_i] + [f"{l}.weight_scale" for l in layers_i])
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight as ck_i8
    for l in layers_i:
        mine = my_int8_convrot_dequant(t[f"{l}.weight"], t[f"{l}.weight_scale"])
        theirs = ck_i8(t[f"{l}.weight"], t[f"{l}.weight_scale"], 256)
        s = stats(mine, theirs)
        s["max_abs_diff_vs_kitchen"] = s.pop("max_abs")
        verify[f"int8:{l}"] = s
        print(f"INT8 decode check {l}: rel_l2={s['rel_l2']:.3e} allclose={s['allclose_1e_5']}")
    save_json(verify, os.path.join(outdir, "independent_decoder_verification.json"))

    # ---------- weight-space parity vs BF16 reference ----------
    proj = [f"model.layers.{i}.{m}" for i in range(36)
            for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]

    bf = get_tensors(BF16, [f"{l}.weight" for l in proj])

    rows_i8, rows_w4 = [], []
    ti = get_tensors(INT8, [f"{l}.weight" for l in proj] + [f"{l}.weight_scale" for l in proj])
    for l in proj:
        w_eff = my_int8_convrot_dequant(ti[f"{l}.weight"], ti[f"{l}.weight_scale"])
        s = stats(w_eff, bf[f"{l}.weight"].float())
        s.update({"layer": l, "kind": "int8_convrot"})
        rows_i8.append(s)

    tw = {}
    CH = 4096
    for i in range(0, len(proj), 7):
        chunk = proj[i:i + 7]
        names = []
        for l in chunk:
            names += [f"{l}.weight", f"{l}.weight_s_rel", f"{l}.weight_s_channel", f"{l}.weight_codebook"]
        tw.update(get_tensors(W4A8, names))
    codebook_is_fixed = 0
    for l in proj:
        cb = tw[f"{l}.weight_codebook"]
        if torch.allclose(cb, torch.tensor(_FIXED_LUT, dtype=cb.dtype), atol=1e-6):
            codebook_is_fixed += 1
        w_eff = my_w4a8_dequant(tw[f"{l}.weight"], tw[f"{l}.weight_s_rel"],
                                tw[f"{l}.weight_s_channel"], cb)
        s = stats(w_eff, bf[f"{l}.weight"].float())
        s.update({"layer": l, "kind": "asym_w4a8_int8"})
        rows_w4.append(s)
    print(f"W4A8 codebook == fixed Gaussian LUT for {codebook_is_fixed}/252 layers")

    for rows, fname in ((rows_i8, "int8_weight_parity.csv"), (rows_w4, "w4a8_weight_parity.csv")):
        fields = ["layer", "kind", "shape", "cosine", "rel_l2", "rmse", "max_abs", "mean_abs", "allclose_1e_5"]
        with open(os.path.join(outdir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in fields} for r in rows])
        rel = [r["rel_l2"] for r in rows]
        print(f"{fname}: n={len(rows)} rel_l2 min={min(rel):.4f} med={sorted(rel)[len(rel)//2]:.4f} max={max(rel):.4f}")
    save_json({"w4a8_codebook_is_fixed_lut": codebook_is_fixed, "total": len(proj)},
              os.path.join(outdir, "w4a8_codebook_provenance.json"))


if __name__ == "__main__":
    main()
