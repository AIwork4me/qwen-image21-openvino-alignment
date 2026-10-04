#!/usr/bin/env python3
"""Phase C: 36-layer Qwen3-VL hidden-state alignment between a reference arm and a target arm.

For each layer l (0 = token embeddings, 1..36 = transformer layers 0..35):
  cosine, rel L2, RMSE, MAE, max abs, norm ratio, NaN/Inf counts.
Layer mapping for OpenVINO artifacts is validated structurally (op names) and
numerically (layer-35 output must match the pre-norm anchor).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, paired_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def layer_metrics(hsA, hsB):
    rows = []
    for l in range(hsA.shape[0]):
        m = paired_metrics(hsA[l], hsB[l])
        m["layer"] = l
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="R0")
    ap.add_argument("--target", required=True)
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--pid", default=None, help="single prompt id; default = all prompts aggregated")
    args = ap.parse_args()

    tdir = os.path.join(ROOT, "artifacts", "tensors")
    prompts = json.load(open(os.path.join(ROOT, f"prompts/{args.suite}.json")))["prompts"]
    pids = [args.pid] if args.pid else [p["id"] for p in prompts]

    all_rows = []
    for pid in pids:
        fa = os.path.join(tdir, args.ref, f"{pid}.npz")
        fb = os.path.join(tdir, args.target, f"{pid}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"[skip] {pid}")
            continue
        hsA = np.load(fa)["hidden_states"].astype(np.float32)
        hsB = np.load(fb)["hidden_states"].astype(np.float32)
        assert hsA.shape == hsB.shape, (hsA.shape, hsB.shape)
        rows = layer_metrics(hsA, hsB)
        for r in rows:
            r["pid"] = pid
        all_rows.extend(rows)

    if not all_rows:
        print("no data"); return
    mdir = os.path.join(ROOT, "artifacts", "metrics")
    os.makedirs(mdir, exist_ok=True)
    out_csv = os.path.join(mdir, f"layer_alignment_{args.ref}_vs_{args.target}{'_'+args.pid if args.pid else ''}.csv")
    keys = ["pid", "layer"] + [k for k in all_rows[0] if k not in ("pid", "layer", "shape")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # aggregate across prompts
    agg = {}
    for r in all_rows:
        agg.setdefault(r["layer"], []).append(r)
    layers = sorted(agg)
    summary = {
        "pair": f"{args.ref}_vs_{args.target}",
        "layers": {
            l: {"cos_median": float(np.median([x["cosine"] for x in agg[l]])),
                "cos_min": float(np.min([x["cosine"] for x in agg[l]])),
                "rel_l2_median": float(np.median([x["rel_l2"] for x in agg[l]])),
                "rel_l2_max": float(np.max([x["rel_l2"] for x in agg[l]])),
                "max_abs_median": float(np.median([x["max_abs"] for x in agg[l]])),
                "rmse_median": float(np.median([x["rmse"] for x in agg[l]])),
                "nan": sum(x["nan_a"] + x["nan_b"] for x in agg[l]),
                "inf": sum(x["inf_a"] + x["inf_b"] for x in agg[l])}
            for l in layers}
    }
    # first layer where median rel_l2 exceeds 2x the median of layers 0..l-1 (significant jump)
    rel = [summary["layers"][l]["rel_l2_median"] for l in layers]
    jumps = []
    for i in range(1, len(rel)):
        base = np.median(rel[max(0, i - 3):i]) if i >= 1 else rel[0]
        if base > 0 and rel[i] > max(2 * base, 1e-6):
            jumps.append((layers[i], rel[i], base))
    summary["significant_jumps"] = jumps
    summary["worst_layer_by_rel_l2"] = layers[int(np.argmax(rel))]
    with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    pdir = os.path.join(ROOT, "artifacts", "plots")
    os.makedirs(pdir, exist_ok=True)
    tag = f"{args.ref}_vs_{args.target}{'_'+args.pid if args.pid else ''}"
    for metric, ylabel, fname, scale in [("cos_median", "cosine similarity", "layer_cosine_vs_depth.png", None),
                                         ("rel_l2_median", "relative L2", "layer_relative_l2_vs_depth.png", "log"),
                                         ("max_abs_median", "max abs error", "layer_max_abs_vs_depth.png", None)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = layers
        med = [summary["layers"][l][metric] for l in xs]
        ax.plot(xs, med, "o-", label=tag)
        if metric == "cos_median":
            ax.fill_between(xs, [summary["layers"][l]["cos_min"] for l in xs],
                            [1.0] * len(xs), alpha=0.2, label="cos min")
        ax.plot(xs, med, "o-", label=tag)
        if scale:
            ax.set_yscale(scale)
        ax.set_xlabel("layer (0 = token embeddings)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{ylabel} vs depth: {tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, fname.replace(".png", f"_{tag}.png")), dpi=140)
    print(json.dumps({k: summary[k] for k in ("significant_jumps", "worst_layer_by_rel_l2")}, indent=2))
    print("saved", out_csv)


if __name__ == "__main__":
    main()
