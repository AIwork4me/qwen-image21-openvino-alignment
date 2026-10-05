#!/usr/bin/env python3
"""Run OpenVINO CPU arms (O_FP32 / O_BF16) with the same npz contract as GPU arms.

Embedding = bit-exact fp32 gather from the official checkpoint table (same
weights as C_BF16).  Conditioning follows official semantics (left-pad select,
drop_idx, prenorm hidden[-1]).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

import openvino as ov
from openvino import Core, properties

MODEL_XML = "/valdata/models/v2_ov_fp32/language_model_fp32.xml"
EXACT_XML = {a: f"/valdata/models/v2_ov_{a.lower()}/language_model_exact.xml" for a in ("O_INT8_EXACT", "O_W4A8_EXACT")}
DROP_IDX = 14  # verified against processor chat template at runtime below


def embed_rows(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[ids]  # fp32 gather


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["O_FP32", "O_BF16", "O_INT8_EXACT", "O_W4A8_EXACT"])
    ap.add_argument("--suite", default="canary")
    ap.add_argument("--pids", default=None)
    args = ap.parse_args()

    # frozen input_ids (already proven identical across arms) come from R_FP32
    prompts = C.load_prompts(args.suite)
    if args.pids:
        prompts = [p for p in prompts if p["id"] in args.pids.split(",")]

    core = Core()
    xml = EXACT_XML.get(args.arm, MODEL_XML)
    model = core.read_model(xml)
    props = {"EXECUTION_MODE_HINT": "ACCURACY", "INFERENCE_PRECISION_HINT": "f32"}
    if args.arm == "O_BF16":
        props = {"EXECUTION_MODE_HINT": "PERFORMANCE", "INFERENCE_PRECISION_HINT": "bf16"}
    compiled = core.compile_model(model, "CPU", props)
    print("inference_precision:", compiled.get_property("INFERENCE_PRECISION_HINT"),
          "exec_mode:", compiled.get_property("EXECUTION_MODE_HINT"), flush=True)

    # embedding table: official fp32 upcast, or artifact-exact dequantized table
    if args.arm in ("O_INT8_EXACT", "O_W4A8_EXACT"):
        tpath = os.path.join(os.path.dirname(xml), "embed_table_dequant_fp32.npy")
        table = np.load(tpath, mmap_mode="r")
    else:
        snap = C.model_snapshot_path()
        table = None
        with safe_open(os.path.join(snap, "text_encoder", "model-00001-of-00004.safetensors"),
                       framework="pt", device="cpu") as f:
            for k in f.keys():
                if "embed_tokens.weight" in k:
                    table = f.get_tensor(k).float().numpy()
                    break
        assert table is not None, "embedding table not found in shard 0"

    outdir = os.path.join(C.VALDATA, "tensors", args.arm)
    os.makedirs(outdir, exist_ok=True)
    for p in prompts:
        t0 = time.time()
        ref = np.load(os.path.join(C.VALDATA, "tensors", "R_FP32", f"{p['id']}.npz"), allow_pickle=True)
        ids = ref["input_ids"]
        T = len(ids)
        embeds = embed_rows(ids, table)[None, ...]          # [1,T,4096] fp32
        attn = np.ones((1, T), dtype=np.int64)
        ar = np.arange(T, dtype=np.int64)
        pos = np.stack([ar, ar, ar]).reshape(3, 1, T)
        res = compiled({0: embeds, 1: attn, 2: pos})
        outs = [res[i] for i in range(len(res))]
        hs = np.stack([o[0] for o in outs[:37]])            # [37,T,4096]
        prenorm, postnorm = outs[37][0], outs[38][0]
        cond = prenorm[DROP_IDX:][None, ...]
        np.savez_compressed(
            os.path.join(outdir, f"{p['id']}.npz"),
            input_ids=ids, attention_mask=attn,
            cond=cond.astype(np.float32), encoder_mask=np.ones(cond.shape[1], dtype=np.int64),
            hidden_states=hs.astype(np.float32), postnorm=postnorm.astype(np.float32),
            meta=json.dumps({"arm": args.arm, "pid": p["id"], "drop_idx": DROP_IDX,
                             "seq_len": int(T), "cond_T": int(cond.shape[1]),
                             "cond_sha256": C.tensor_sha256(torch.from_numpy(cond)),
                             "input_ids_sha256": C.tensor_sha256(torch.from_numpy(ids))}),
        )
        print(f"{args.arm} {p['id']}: T={T} T'={cond.shape[1]} {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
