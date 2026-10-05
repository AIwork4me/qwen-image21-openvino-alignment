#!/usr/bin/env python3
"""v2 CPU/GPU prompt-length regression: token counts 16..512 must not fail.

Arms: O_FP32, O_BF16, O_INT8_EXACT, O_W4A8_EXACT (OpenVINO CPU) and the
actual customer artifacts re-run through the ComfyUI loader (C_INT8, C_W4A8)
on GPU where practical.

For each (arm, target_len): synthesize a token sequence of exactly that
length from the tokenizer (repeated noun phrase, EOS-terminated), record
PASS/FAIL, latency, peak RSS, output shape, NaN/Inf, error message.

Outputs: artifacts/v2/metrics/runtime_prompt_length_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

LENGTHS = [16, 32, 40, 48, 64, 96, 128, 171, 256, 512]


def synth_ids(tokenizer, n: int) -> np.ndarray:
    ids = tokenizer("apple banana cherry dolphin elephant ", return_tensors=None)["input_ids"]
    unit = [i for i in ids if i not in (tokenizer.eos_token_id,)]
    out = []
    while len(out) < n - 1:
        out.extend(unit)
    out = out[: n - 1]
    out.append(tokenizer.eos_token_id)
    return np.array(out, dtype=np.int64)


def run_ov(arm: str, ids: np.ndarray) -> dict:
    import openvino as ov
    from openvino import Core
    xml = (f"/valdata/models/v2_ov_{arm.lower()}/language_model_exact.xml"
           if arm != "O_FP32" else "/valdata/models/v2_ov_fp32/language_model_fp32.xml")
    core = Core()
    model = core.read_model(xml)
    props = {"EXECUTION_MODE_HINT": "ACCURACY", "INFERENCE_PRECISION_HINT": "f32"}
    if arm == "O_BF16":
        props = {"EXECUTION_MODE_HINT": "PERFORMANCE", "INFERENCE_PRECISION_HINT": "bf16"}
    compiled = core.compile_model(model, "CPU", props)
    if arm in ("O_INT8_EXACT", "O_W4A8_EXACT"):
        table = np.load(os.path.join(os.path.dirname(xml), "embed_table_dequant_fp32.npy"), mmap_mode="r")
    else:
        from safetensors import safe_open
        snap = C.model_snapshot_path()
        table = None
        with safe_open(os.path.join(snap, "text_encoder", "model-00001-of-00004.safetensors"),
                       framework="pt", device="cpu") as f:
            for k in f.keys():
                if "embed_tokens.weight" in k:
                    table = f.get_tensor(k).float().numpy()
                    break
    embeds = table[ids][None, ...]
    T = len(ids)
    attn = np.ones((1, T), dtype=np.int64)
    ar = np.arange(T, dtype=np.int64)
    pos = np.stack([ar, ar, ar]).reshape(3, 1, T)
    t0 = time.time()
    res = compiled({0: embeds, 1: attn, 2: pos})
    dt = time.time() - t0
    prenorm = res[37][0]
    return {"latency_s": round(dt, 3),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "out_shape": list(prenorm.shape),
            "nan": bool(np.isnan(prenorm).any()), "inf": bool(np.isinf(prenorm).any())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="O_FP32,O_BF16,O_INT8_EXACT,O_W4A8_EXACT")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    snap = C.model_snapshot_path()
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)

    rows = []
    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    out = os.path.join(mdir, "runtime_prompt_length_matrix.csv")
    for arm in args.arms.split(","):
        for n in LENGTHS:
            ids = synth_ids(tok, n)
            rec = {"arm": arm, "target_len": n, "actual_len": len(ids)}
            try:
                rec.update(run_ov(arm, ids))
                rec["status"] = "PASS" if not (rec["nan"] or rec["inf"]) else "FAIL_NONFINITE"
            except Exception as e:  # noqa: BLE001
                rec.update({"status": "FAIL", "error": str(e)[:200]})
            rows.append(rec)
            print(rec, flush=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
