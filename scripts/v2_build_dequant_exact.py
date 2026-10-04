#!/usr/bin/env python3
"""v2: build dequantized-equivalent HF checkpoints of the official ComfyUI quant artifacts.

For each quantized tensor in qwen3vl_8b_int8_convrot / qwen3vl_8b_w4a8:
  effective weight = independent spec decoder (bit-exact vs comfy_kitchen, verified)
  -> cast to bf16 (exactly what the stock ComfyUI conditioning path multiplies)
Non-quantized tensors pass through unchanged.
Key map: comfy 'model.X' -> HF 'model.language_model.X'; lm_head unchanged.

Output: a full HF text_encoder dir (config/tokenizer copied from the official
snapshot) whose weights are the effective dequantized weights. This is the
'EXACT' OpenVINO arm input: identical effective weights, no requantization.
"""
import json
import os
import shutil
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, save_json, sha256_file
from v2_weight_parity import (BF16, INT8, W4A8, get_tensors, my_int8_convrot_dequant,
                              my_w4a8_dequant, hdr_of)

OUTROOT = "/valdata/models/v2_dequant_exact"


def dequant_artifact(src, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    hdr, _ = hdr_of(src)
    comfy_quant_layers = {k[:-len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant")}
    n_i8 = n_w4 = 0
    out_sd = {}
    # collect all storage names
    plain = [k for k in hdr if k != "__metadata__" and not k.endswith(
        (".comfy_quant", ".weight_scale", ".weight_s_rel", ".weight_s_channel", ".weight_codebook"))]
    quant_names = sorted(comfy_quant_layers)
    print(f"[{tag}] {len(quant_names)} quantized layers, {len(plain)} plain tensors")

    from safetensors.torch import save_file
    # process plain tensors (bf16 passthrough)
    for i in range(0, len(plain), 500):
        chunk = plain[i:i + 500]
        t = get_tensors(src, chunk)
        for k, v in t.items():
            if k in comfy_quant_layers:
                continue
            out_sd[k] = v
    # quantized layers, in chunks to bound RAM
    for i in range(0, len(quant_names), 7):
        chunk = quant_names[i:i + 7]
        names = []
        for l in chunk:
            h = hdr
            names.append(f"{l}.weight")
            for suf in (".weight_scale", ".weight_s_rel", ".weight_s_channel", ".weight_codebook"):
                if f"{l}{suf}" in h:
                    names.append(f"{l}{suf}")
        t = get_tensors(src, names)
        for l in chunk:
            cq = get_tensors(src, [f"{l}.comfy_quant"])[f"{l}.comfy_quant"]
            conf = json.loads(cq.numpy().tobytes().decode())
            if conf["format"] == "int8_tensorwise":
                w = my_int8_convrot_dequant(t[f"{l}.weight"], t[f"{l}.weight_scale"])
                n_i8 += 1
            elif conf["format"] == "asym_w4a8_int8":
                w = my_w4a8_dequant(t[f"{l}.weight"], t[f"{l}.weight_s_rel"],
                                    t[f"{l}.weight_s_channel"], t[f"{l}.weight_codebook"])
                n_w4 += 1
            else:
                raise ValueError(conf)
            out_sd[l] = w.to(torch.bfloat16)
        del t
        if (i // 7) % 24 == 0:
            print(f"[{tag}] {i + len(chunk)}/{len(quant_names)} layers dequantized", flush=True)

    # rename remaining plain keys to HF layout
    out_sd = { (k.replace("model.", "model.language_model.", 1)
                if k.startswith("model.") else k): v for k, v in out_sd.items() }
    path = os.path.join(out_dir, "model.safetensors")
    print(f"[{tag}] saving {path} with {len(out_sd)} tensors "
          f"(int8: {n_i8}, w4: {n_w4})", flush=True)
    from safetensors.torch import save_file
    save_file(out_sd, path, metadata={"format": "pt"})
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": sha256_file(path), "n_int8_layers": n_i8, "n_w4_layers": n_w4}


def main():
    cfg = load_cfg()
    snap_text = os.path.join(cfg["model"]["snapshot_path"], "text_encoder")
    manifest = {}
    for tag, src in (("O_INT8_EXACT_src", INT8), ("O_W4A8_EXACT_src", W4A8)):
        out_dir = os.path.join(OUTROOT, tag.replace("_src", ""))
        if not os.path.exists(os.path.join(out_dir, "model.safetensors")):
            os.makedirs(out_dir, exist_ok=True)
            for f in os.listdir(snap_text):
                if f.endswith((".json", ".txt", ".jinja")) and f != "model.safetensors.index.json":
                    shutil.copy(os.path.join(snap_text, f), os.path.join(out_dir, f))
            manifest[tag] = dequant_artifact(src, out_dir, tag)
        else:
            print(f"[{tag}] already exists")
    save_json(manifest, os.path.join("artifacts", "v2", "manifests", "dequant_exact_checkpoints.json"))


if __name__ == "__main__":
    main()
