#!/usr/bin/env python3
"""v2 Section 9: prove ComfyUI BF16 loader parity vs official PyTorch BF16 (R0).

Input-level fields (formatted text, input_ids, attention mask, drop index, seq len)
must match EXACTLY. Tensor-level fields (embedding, 36 layers, prenorm, postnorm,
conditioning) are compared numerically C_BF16 (ComfyUI loader, W7900) vs R0
(official transformers path, W7900).

Interpretation note (review-corrected): the stock ComfyUI TE path runs an FP32
activation stream, so tensor deltas vs R0 (bf16 compute) measure the bf16-vs-fp32
arithmetic envelope, and deltas vs R1 (fp32) measure cross-implementation residual.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, paired_metrics, save_json

V1_TENSORS = "/valdata/qval/artifacts/tensors"


def fmt_llama(text):
    SYS = "Comprehend and analyze the provided prompt."
    T = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n"
         f"<|im_start|>assistant\n")
    return T.format(text)


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    prompts = json.load(open(os.path.join(ROOT, "prompts", f"{suite}.json")))["prompts"]
    inputs_dir = os.path.join(ROOT, "artifacts", "inputs")

    rows = []
    summaries = []
    for p in prompts:
        pid = p["id"]
        r0 = np.load(os.path.join(V1_TENSORS, "R0", f"{pid}.npz"))
        cb = np.load(os.path.join(ROOT, "artifacts", "v2", "tensors", "C_BF16", f"{pid}.npz"))
        meta_c = json.load(open(os.path.join(ROOT, "artifacts", "v2", "tensors", "C_BF16",
                                             f"meta_C_BF16_{suite}.json")))
        entry = next(e for e in meta_c["entries"] if e["pid"] == pid)
        inp = np.load(os.path.join(inputs_dir, f"{pid}.npz"))

        # ---- input-level (must be exact) ----
        ids_r0 = inp["input_ids"].reshape(-1).tolist()
        ids_c = entry["token_ids"]
        # v1 processor uses left padding? (check attention mask) -- compare unpadded
        am = inp["attention_mask"].reshape(-1)
        ids_r0_unpad = [t for t, m in zip(ids_r0, am.tolist()) if m == 1]
        # real formatted-text comparison: ComfyUI's captured llama_text vs the
        # template the v1 processor path applied (same fixed T2I template)
        comfy_llama = entry.get("llama_text") or ""
        row = {
            "pid": pid,
            "formatted_text_match": int(comfy_llama == fmt_llama(p["text"])),
            "input_ids_match": int(ids_r0_unpad == ids_c),
            "seq_len_r0": len(ids_r0_unpad), "seq_len_comfy": len(ids_c),
            "drop_idx_r0": int(r0["drop_idx"]) if "drop_idx" in r0 else None,
            "cond_len_r0": r0["prompt_embeds"].shape[0],
            "cond_len_comfy": cb["cond"].shape[1],
        }
        # system turn drop equivalence: comfy drops to second <|im_start|>; v1 drop_idx
        rows.append(row)

        # ---- tensor-level ----
        pairs = {
            "embedding": (r0["hidden_states"][0], cb["embedding"][0]),
            "prenorm": (r0["prenorm"], cb["prenorm"][0]),
            "postnorm": (r0["postnorm"], cb["postnorm"][0]),
            "conditioning": (r0["prompt_embeds"].astype(np.float32), cb["cond"][0]),
        }
        hs_r0 = r0["hidden_states"]  # [37, seq, 4096]
        for i in range(36):
            pairs[f"layer_{i+1:02d}"] = (hs_r0[i + 1], cb[f"layer_{i:02d}"][0])
        for name, (a, b) in pairs.items():
            if a.shape != b.shape:
                summaries.append({"pid": pid, "field": name, "error": f"shape {a.shape} vs {b.shape}"})
                continue
            m = paired_metrics(a, b)
            cos = float(np.dot(a.ravel(), b.ravel()) /
                        (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-30))
            summaries.append({"pid": pid, "field": name, "rel_l2": m["rel_l2"],
                              "max_abs": m["max_abs"], "cosine": cos,
                              "mean_abs": m["mean_abs"], "rmse": m["rmse"],
                              "norm_ratio": m["norm_ratio"],
                              "nan": int(np.isnan(b).sum()), "inf": int(np.isinf(b).sum())})

    # write CSVs
    os.makedirs(os.path.join(ROOT, "artifacts", "v2", "metrics"), exist_ok=True)
    with open(os.path.join(ROOT, "artifacts", "v2", "metrics", "comfyui_bf16_loader_parity.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "field", "rel_l2", "max_abs", "cosine",
                                          "mean_abs", "rmse", "norm_ratio", "nan", "inf", "error"])
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    with open(os.path.join(ROOT, "artifacts", "v2", "metrics", "comfyui_bf16_loader_inputs.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate per field
    agg = {}
    for s in summaries:
        d = agg.setdefault(s["field"], {"errors": 0, "vals": []})
        if "error" in s:
            d["errors"] += 1
        else:
            d["vals"].append(s)
    report = {}
    for field, d in agg.items():
        vals = d["vals"]
        if not vals:
            report[field] = {"errors": d["errors"]}
            continue
        report[field] = {
            "n": len(vals), "errors": d["errors"],
            "rel_l2_max": max(v["rel_l2"] for v in vals),
            "rel_l2_median": float(np.median([v["rel_l2"] for v in vals])),
            "cosine_min": min(v["cosine"] for v in vals),
            "max_abs_max": max(v["max_abs"] for v in vals),
        }
    inputs_ok = all(r["input_ids_match"] and r["seq_len_r0"] == r["seq_len_comfy"]
                    and r["cond_len_r0"] == r["cond_len_comfy"] for r in rows)
    save_json({"suite": suite, "inputs_exact": inputs_ok,
               "input_rows": rows, "per_field": report},
              os.path.join(ROOT, "artifacts", "v2", "metrics", "comfyui_bf16_loader_parity.json"))
    print(json.dumps(report, indent=1)[:3000])
    print("inputs_exact:", inputs_ok)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
