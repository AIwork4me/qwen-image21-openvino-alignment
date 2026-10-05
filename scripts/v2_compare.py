#!/usr/bin/env python3
"""v2 core numerical matrix: conditioning pairwise metrics + layer alignment.

Consumes frozen tensors from /valdata/tensors/{arm}/{pid}.npz and emits:
  artifacts/v2/metrics/encoder_matrix_summary.csv   (pairwise cond metrics)
  artifacts/v2/metrics/layer_alignment_{a}_{b}.csv  (per-layer rel_l2/cos/rmse)
  artifacts/v2/metrics/input_identity.json          (input_ids exact-match grid)
  artifacts/v2/plots/layer_*.png
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

DEFAULT_PAIRS = [
    ("R_FP32", "R_BF16"), ("R_BF16", "C_BF16"), ("R_FP32", "C_BF16"),
    ("R_FP32", "O_FP32"), ("R_BF16", "O_BF16"),
    ("C_BF16", "C_INT8"), ("C_BF16", "C_W4A8"), ("C_INT8", "C_W4A8"),
    ("C_INT8", "O_INT8_EXACT"), ("C_INT8", "O_INT8_ALT"),
    ("C_W4A8", "O_W4A8_EXACT"), ("C_W4A8", "O_W4_ALT"),
    ("O_FP32", "O_BF16"), ("O_FP32", "O_INT8_ALT"), ("O_FP32", "O_W4_ALT"),
]


def load_arm_pid(arm: str, pid: str):
    path = os.path.join(C.VALDATA, "tensors", arm, f"{pid}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="canary")
    ap.add_argument("--pairs", default=None, help="semicolon list of A:B pairs")
    ap.add_argument("--layer-pairs", default="C_BF16:C_INT8;C_BF16:C_W4A8;R_FP32:C_BF16;R_FP32:O_FP32;C_INT8:C_INT8_ALT;C_W4A8:C_W4_ALT")
    args = ap.parse_args()

    prompts = C.load_prompts(args.suite)
    pids = [p["id"] for p in prompts]
    pairs = ([tuple(p.split(":")) for p in args.pairs.split(";")]
             if args.pairs else DEFAULT_PAIRS)

    available = {a for a in C.ARMS for pid in pids if load_arm_pid(a, pid)}
    pairs = [(a, b) for a, b in pairs if a in available and b in available]
    print("arms with tensors:", sorted(available))

    # ---- input identity grid
    identity = {}
    ref_ids = None
    for pid in pids:
        for arm in sorted(available):
            z = load_arm_pid(arm, pid)
            if z is None:
                continue
            entry = identity.setdefault(pid, {})
            entry[arm] = {"ids_sha256": C.tensor_sha256(torch.from_numpy(z["input_ids"])),
                          "seq_len": int(len(z["input_ids"]))}
        base = "R_FP32" if "R_FP32" in identity[pid] else sorted(identity[pid])[0]
        identity[pid]["all_exact_vs_" + base] = all(
            v["ids_sha256"] == identity[pid][base]["ids_sha256"] for k, v in identity[pid].items() if isinstance(v, dict))
    C.save_json(identity, os.path.join(C.REPO, "artifacts", "v2", "metrics", "input_identity.json"))

    # ---- pairwise conditioning metrics
    rows = []
    for a, b in pairs:
        for pid in pids:
            za, zb = load_arm_pid(a, pid), load_arm_pid(b, pid)
            if za is None or zb is None:
                continue
            ca = torch.from_numpy(za["cond"]).unsqueeze(0) if za["cond"].ndim == 2 else torch.from_numpy(za["cond"])
            cb = torch.from_numpy(zb["cond"]).unsqueeze(0) if zb["cond"].ndim == 2 else torch.from_numpy(zb["cond"])
            if ca.shape != cb.shape:
                rows.append({"arm_a": a, "arm_b": b, "pid": pid,
                             "shape_mismatch": f"{list(ca.shape)}!={list(cb.shape)}"})
                continue
            m = C.pair_metrics(ca, cb)
            m.update({"arm_a": a, "arm_b": b, "pid": pid})
            rows.append({k: m.get(k) for k in ("arm_a", "arm_b", "pid", "rel_l2", "cosine", "rmse",
                                               "mean_abs_err", "median_abs_err", "max_abs_err",
                                               "per_token_cosine_min", "per_token_rel_l2_max",
                                               "norm_ratio", "sign_agreement", "nan", "inf",
                                               "shape_mismatch")})
    out = os.path.join(C.REPO, "artifacts", "v2", "metrics", "encoder_matrix_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out} ({len(rows)} rows)")
    for a, b in pairs:
        sel = [r for r in rows if r["arm_a"] == a and r["arm_b"] == b and r.get("rel_l2") is not None]
        if sel:
            s = C.stats_block([r["rel_l2"] for r in sel])
            print(f"  {a:10s}:{b:10s} n={s['n']:3d} rel_l2 median={s['median']:.5f} "
                  f"p5={s['p5']:.5f} p95={s['p95']:.5f}")

    # ---- layer alignment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layer_pairs = [tuple(p.split(":")) for p in args.layer_pairs.split(";") if ":" in p]
    for a, b in layer_pairs:
        if a not in available or b not in available:
            continue
        recs = []
        for pid in pids:
            za, zb = load_arm_pid(a, pid), load_arm_pid(b, pid)
            if za is None or zb is None:
                continue
            ha, hb = torch.from_numpy(za["hidden_states"]), torch.from_numpy(zb["hidden_states"])
            n = min(ha.shape[0], hb.shape[0])
            for i in range(n):
                label = "embed" if i == 0 else f"layer_{i}"
                recs.append({"pid": pid, "position": i, "label": label,
                             "rel_l2": C.rel_l2(ha[i], hb[i]),
                             "cosine": C.cosine(ha[i], hb[i]),
                             "rmse": float((ha[i].double() - hb[i].double()).pow(2).mean().sqrt())})
        if not recs:
            continue
        out = os.path.join(C.REPO, "artifacts", "v2", "metrics", f"layer_alignment_{a}_{b}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        # aggregate + plot
        agg = {}
        for r in recs:
            agg.setdefault(r["position"], []).append(r["rel_l2"])
        xs = sorted(agg)
        med = [float(np.median(agg[x])) for x in xs]
        p95 = [float(np.percentile(agg[x], 95)) for x in xs]
        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, med, marker="o", ms=3, label="median rel L2")
        plt.plot(xs, p95, marker="^", ms=3, label="p95 rel L2")
        plt.yscale("log")
        plt.xlabel("position (0=embedding, 1..36=decoder layers)")
        plt.ylabel("rel L2")
        plt.title(f"layer divergence {a} vs {b} ({args.suite})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot = os.path.join(C.REPO, "artifacts", "v2", "plots", f"layer_{a}_{b}.png")
        os.makedirs(os.path.dirname(plot), exist_ok=True)
        plt.savefig(plot, dpi=120)
        plt.close()
        print(f"-> {out} + plot ({len(recs)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
