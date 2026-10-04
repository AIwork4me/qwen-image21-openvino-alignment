#!/usr/bin/env python3
"""v2 Section 13/14: embedding matrix + 36-layer attribution across all v2 arms.

Comparisons (pair names stable):
  R_FP32 <-> O_FP32     (v1 evidence reuse, verified)
  R_BF16 <-> C_BF16     (loader parity, canary)
  R_FP32 <-> R_BF16     (bf16 envelope)
  C_BF16 <-> C_INT8     (int8 error)
  C_BF16 <-> C_W4A8     (w4a8 error)
  R_BF16 <-> C_INT8 / C_W4A8 (cross)
  repeatability arms (*_rep r0 vs r1 vs r2)

Outputs: encoder_matrix_summary.csv, layer_error_*.csv
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, paired_metrics, save_json

V1 = "/valdata/qval/artifacts/tensors"
V2 = os.path.join(ROOT, "artifacts", "v2", "tensors")

# arm alias -> (v1_dir | v2_dir, file pattern)
ARM_FILES = {
    "R_FP32": (V1, "R1"), "R_BF16": (V1, "R0"),
    "O_FP32": (V1, "O0"), "O_BF16": (V1, "O1"),
}


def load_arm(arm, pid, suite):
    """Return dict with embedding, layers 1..36, prenorm, postnorm, cond."""
    if arm in ARM_FILES:
        base, d = ARM_FILES[arm]
        z = np.load(os.path.join(base, d, f"{pid}.npz"))
        hs = z["hidden_states"]
        return {"embedding": hs[0], **{f"layer_{i+1:02d}": hs[i + 1] for i in range(36)},
                "prenorm": z["prenorm"], "postnorm": z["postnorm"],
                "cond": z["prompt_embeds"].astype(np.float32)}
    if "_rep_r" in arm:  # e.g. C_INT8_rep_r0 -> dir C_INT8_rep, file C01_r0.npz
        d, rep = arm.split("_rep_r")
        z = np.load(os.path.join(V2, f"{d}_rep", f"{pid}_r{rep}.npz"))
    else:
        z = np.load(os.path.join(V2, arm, f"{pid}.npz"))
    return {"embedding": z["embedding"][0],
            **{f"layer_{i+1:02d}": z[f"layer_{i:02d}"][0] for i in range(36)},
            "prenorm": z["prenorm"][0], "postnorm": z["postnorm"][0], "cond": z["cond"][0]}


def pair_stats(a, b):
    m = paired_metrics(a, b)
    cos = float(np.dot(a.ravel(), b.ravel()) /
                (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-30))
    # per-token cosine
    at, bt = a.reshape(-1, a.shape[-1]).astype(np.float64), b.reshape(-1, b.shape[-1]).astype(np.float64)
    tok_cos = (at * bt).sum(1) / (np.linalg.norm(at, axis=1) * np.linalg.norm(bt, axis=1) + 1e-30)
    m.update({
        "cosine": cos,
        "per_token_cos_min": float(tok_cos.min()), "per_token_cos_mean": float(tok_cos.mean()),
        "nan": int(np.isnan(b).sum()), "inf": int(np.isinf(b).sum()),
        "sign_agree": float((np.sign(a) == np.sign(b)).mean()),
    })
    return m


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "canary"
    pids = [p["id"] for p in json.load(open(os.path.join(ROOT, "prompts",
                                                         f"{suite}.json")))["prompts"]]

    pair_defs = [
        ("R_FP32", "O_FP32"), ("R_BF16", "C_BF16"), ("R_FP32", "R_BF16"),
        ("C_BF16", "C_INT8"), ("C_BF16", "C_W4A8"),
        ("R_BF16", "C_INT8"), ("R_BF16", "C_W4A8"),
        ("R_FP32", "O_BF16"), ("C_BF16", "O_BF16"), ("C_BF16", "O_FP32"),
    ]
    # repeatability
    rep_defs = [(f"{a}_rep_r{i}", f"{a}_rep_r{j}") for a in ("C_BF16", "C_INT8", "C_W4A8")
                for i, j in ((0, 1), (1, 2))]

    os.makedirs(os.path.join(ROOT, "artifacts", "v2", "metrics"), exist_ok=True)
    summary_rows, layer_rows = [], []
    layer_fields = ["pair", "pid", "field", "rel_l2", "rmse", "max_abs", "cosine", "norm_ratio", "nan", "inf"]

    def compare(name_a, name_b, arm_a, arm_b, layer_file=False):
        agg = {}
        for pid in pids:
            try:
                A, B = load_arm(arm_a, pid, suite), load_arm(arm_b, pid, suite)
            except FileNotFoundError as e:
                print(f"skip {name_a}:{name_b} {pid}: {e}")
                return
            for field in ["cond"]:
                s = pair_stats(A[field], B[field])
                s.update({"pair": f"{name_a}:{name_b}", "pid": pid, "field": field})
                summary_rows.append(s)
                agg.setdefault(field, []).append(s)
            if layer_file:
                for i in range(36):
                    f = f"layer_{i+1:02d}"
                    s = pair_stats(A[f], B[f])
                    layer_rows.append({"pair": f"{name_a}:{name_b}", "pid": pid, "field": f,
                                       "rel_l2": s["rel_l2"], "rmse": s["rmse"],
                                       "max_abs": s["max_abs"], "cosine": s["cosine"],
                                       "norm_ratio": s["norm_ratio"], "nan": s["nan"],
                                       "inf": s["inf"]})
                for f in ("embedding", "prenorm", "postnorm"):
                    s = pair_stats(A[f], B[f])
                    layer_rows.append({"pair": f"{name_a}:{name_b}", "pid": pid, "field": f,
                                       "rel_l2": s["rel_l2"], "rmse": s["rmse"],
                                       "max_abs": s["max_abs"], "cosine": s["cosine"],
                                       "norm_ratio": s["norm_ratio"], "nan": s["nan"],
                                       "inf": s["inf"]})
        print(f"{name_a}:{name_b} " + " ".join(
            f"{f}: relL2 med={np.median([x['rel_l2'] for x in v]):.4g} "
            f"max={np.max([x['rel_l2'] for x in v]):.4g} cos_min={np.min([x['cosine'] for x in v]):.6f}"
            for f, v in agg.items()))

    # straight pairs (arm names now resolve via ARM_FILES/load_arm)
    for a, b in pair_defs:
        layer = (a, b) in (("C_BF16", "C_INT8"), ("C_BF16", "C_W4A8"),
                           ("C_BF16", "O_BF16"))
        compare(a, b, a, b, layer_file=layer)

    for a, b in rep_defs:
        arm_a = a.replace("_r0", "").replace("_r1", "").replace("_r2", "")
        arm_b = b.replace("_r0", "").replace("_r1", "").replace("_r2", "")
        compare(a, b, a, b, layer_file=False)

    with open(os.path.join(ROOT, "artifacts", "v2", "metrics", "encoder_matrix_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    if layer_rows:
        for tag, fname in ((("C_BF16", "C_INT8"), "layer_error_int8.csv"),
                           (("C_BF16", "C_W4A8"), "layer_error_w4a8.csv")):
            rows = [r for r in layer_rows if r["pair"] == f"{tag[0]}:{tag[1]}"]
            with open(os.path.join(ROOT, "artifacts", "v2", "metrics", fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=layer_fields)
                w.writeheader()
                w.writerows(rows)
    print("wrote encoder_matrix_summary.csv",
          f"({len(summary_rows)} rows) + layer files ({len(layer_rows)} rows)")


if __name__ == "__main__":
    main()
