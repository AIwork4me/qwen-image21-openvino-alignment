#!/usr/bin/env python3
"""v2 performance matrix: artifact size, load time, cold/warm latency, RSS/VRAM
per encoder arm, plus conditioning error + (later joined) quality columns.

Arms measured end-to-end on the canary prompt C09 (representative length):
  GPU: R_FP32, R_BF16, C_BF16, C_INT8, C_W4A8 (ComfyUI loader, warm x N)
  CPU: O_FP32, O_BF16, O_INT8_EXACT, O_W4A8_EXACT (warm x N)

Outputs: artifacts/v2/metrics/perf_matrix.csv (+ rss samples), Pareto inputs.
Accuracy arms are never tuned for performance here (separate measurement).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C


def gpu_vram_mb() -> int:
    return int(torch.cuda.memory_allocated() // (1024 * 1024))


def measure_gpu(arm: str, pid: str, warm: int) -> dict:
    import importlib
    import v2_run_encoder_arm as runner
    importlib.reload(runner)
    t0 = time.time()
    if arm.startswith("R_"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(os.path.join(runner.SNAP, "processor"), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(runner.SNAP, "text_encoder"),
            torch_dtype=torch.float32 if arm == "R_FP32" else torch.bfloat16,
            device_map="cuda", local_files_only=True).eval()
    else:
        clip = runner.load_comfy_clip(C.ENCODER_FILES[arm])
        clip.load_model()
        tokenizr = clip.tokenizer
    load_s = time.time() - t0

    prompt = [p for p in C.load_prompts("canary") if p["id"] == pid][0]
    text = prompt["text"]
    lat = []
    if arm.startswith("R_"):
        template = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        for _ in range(warm + 1):
            mi = processor(text=[template.format(text)], padding=True,
                           padding_side="left", return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                model(input_ids=mi.input_ids, attention_mask=mi.attention_mask)
            torch.cuda.synchronize(); lat.append(time.time() - t0)
        size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        del model
    else:
        tokens = tokenizr.tokenize_with_weights(text)
        for _ in range(warm + 1):
            torch.cuda.synchronize(); t0 = time.time()
            clip.cond_stage_model.encode_token_weights(tokens)
            torch.cuda.synchronize(); lat.append(time.time() - t0)
        size_gb = os.path.getsize(os.path.join(C.ENCODER_DIR, C.ENCODER_FILES[arm])) / 1e9
        del clip
    torch.cuda.empty_cache()
    return {"artifact_size_gb": round(size_gb, 3), "load_s": round(load_s, 1),
            "cold_s": round(lat[0], 3),
            "warm_p50_s": round(statistics.median(lat[1:]), 3),
            "warm_p90_s": round(sorted(lat[1:])[int(0.9 * (len(lat) - 1))], 3),
            "warm_p95_s": round(sorted(lat[1:])[int(0.95 * (len(lat) - 1))], 3),
            "n_warm": warm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-arms", default="R_BF16,C_BF16,C_INT8,C_W4A8")
    ap.add_argument("--cpu-arms", default="O_FP32,O_BF16,O_INT8_EXACT,O_W4A8_EXACT")
    ap.add_argument("--pid", default="C09")
    ap.add_argument("--warm", type=int, default=9)
    args = ap.parse_args()
    C.gate_environment()

    rows = []
    for arm in args.gpu_arms.split(","):
        rec = {"arm": arm, "device": "gpu"}
        rec.update(measure_gpu(arm, args.pid, args.warm))
        rows.append(rec)
        print(rec, flush=True)

    # CPU arms via subprocess isolation (fair RSS measurement)
    for arm in args.cpu_arms.split(","):
        code = f'''
import sys, os, time, resource, statistics, json
sys.path.insert(0, "{C.REPO}/scripts")
import numpy as np, torch
import v2_common as C
import v2_run_ov_arm as R
import openvino as ov
from openvino import Core
from safetensors import safe_open
arm = "{arm}"
xml = "/valdata/models/v2_ov_fp32/language_model_fp32.xml" if arm=="O_FP32" else f"/valdata/models/v2_ov_{{arm.lower()}}/language_model_exact.xml"
t0=time.time()
core=Core(); model=core.read_model(xml)
props={{"EXECUTION_MODE_HINT":"ACCURACY","INFERENCE_PRECISION_HINT":"f32"}}
if arm=="O_BF16": props={{"EXECUTION_MODE_HINT":"PERFORMANCE","INFERENCE_PRECISION_HINT":"bf16"}}
compiled=core.compile_model(model,"CPU",props)
load_s=time.time()-t0
if arm in ("O_INT8_EXACT","O_W4A8_EXACT"):
    table=np.load(os.path.join(os.path.dirname(xml),"embed_table_dequant_fp32.npy"),mmap_mode="r")
else:
    snap=C.model_snapshot_path(); table=None
    with safe_open(os.path.join(snap,"text_encoder","model-00001-of-00004.safetensors"),framework="pt",device="cpu") as f:
        for k in f.keys():
            if "embed_tokens.weight" in k: table=f.get_tensor(k).float().numpy(); break
ref=np.load(os.path.join(C.VALDATA,"tensors","R_FP32","{args.pid}.npz"),allow_pickle=True)
ids=ref["input_ids"]; T=len(ids); embeds=table[ids][None,...]
attn=np.ones((1,T),dtype=np.int64); ar=np.arange(T,dtype=np.int64); pos=np.stack([ar,ar,ar]).reshape(3,1,T)
lat=[]
for i in range({args.warm + 1}):
    t0=time.time(); r=compiled({{0:embeds,1:attn,2:pos}}); lat.append(time.time()-t0)
size_gb=os.path.getsize(xml)/1e9 + os.path.getsize(xml.replace(".xml",".bin"))/1e9
if arm in ("O_INT8_EXACT","O_W4A8_EXACT"):
    size_gb += os.path.getsize(os.path.join(os.path.dirname(xml),"embed_table_dequant_fp32.npy"))/1e9
print(json.dumps({{"artifact_size_gb":round(size_gb,3),"load_s":round(load_s,1),"cold_s":round(lat[0],3),
 "warm_p50_s":round(statistics.median(lat[1:]),3),"warm_p90_s":round(sorted(lat[1:])[int(0.9*(len(lat)-1))],3),
 "warm_p95_s":round(sorted(lat[1:])[int(0.95*(len(lat)-1))],3),"n_warm":{args.warm},
 "peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}}))
'''
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        rec = {"arm": arm, "device": "cpu"}
        try:
            rec.update(json.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:
            rec["error"] = out.stderr[-300:]
        rows.append(rec)
        print(rec, flush=True)

    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    out_csv = os.path.join(mdir, "perf_matrix.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
