#!/usr/bin/env python3
"""Phase B: final embedding numerical alignment between encoder arms.

Compares pairs of arms on: prenorm (full-seq conditioning tensor), prompt_embeds
(kept tokens after drop_idx), token embeddings. Emits summary CSV + per-token CSV + plots.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, paired_metrics, per_token_cosine, per_token_summary, bf16_exact_match, stats_block

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_arm(arm, pid, key):
    p = os.path.join(ROOT, "artifacts", "tensors", arm, f"{pid}.npz")
    z = np.load(p)
    return z[key].astype(np.float32), int(z["drop_idx"])


def compare_pair(armA, armB, pid):
    row = {"pair": f"{armA}_vs_{armB}", "pid": pid}
    for key in ("prenorm", "prompt_embeds"):
        a, dA = load_arm(armA, pid, key)
        b, dB = load_arm(armB, pid, key)
        if key == "prenorm":
            m = paired_metrics(a, b)
            row.update({f"{key}.{k}": v for k, v in m.items()})
            row[f"{key}.bf16_exact_match"] = bf16_exact_match(a, b)
        else:
            m = paired_metrics(a, b)
            row.update({f"{key}.{k}": v for k, v in m.items()})
            row[f"{key}.bf16_exact_match"] = bf16_exact_match(a, b)
            c = per_token_cosine(a, b)
            row[f"{key}.per_token_cos"] = json.dumps(per_token_summary(c))
    # token embeddings (OV arms store token_embeddings; ROCm stores in hidden_states[0])
    try:
        ea, _ = load_arm(armA, pid, "token_embeddings")
    except KeyError:
        ea, _ = load_arm(armA, pid, "hidden_states")
        ea = ea[0]
    try:
        eb, _ = load_arm(armB, pid, "token_embeddings")
    except KeyError:
        eb, _ = load_arm(armB, pid, "hidden_states")
        eb = eb[0]
    m = paired_metrics(ea, eb)
    row.update({f"embeds.{k}": v for k, v in m.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["R0:R0R", "R0:R1", "R0:O0", "O0:O1", "O0:O3", "R0:O3", "R0:O1F", "O0:O1F"])
    ap.add_argument("--suite", default="smoke")
    args = ap.parse_args()

    prompts = json.load(open(os.path.join(ROOT, f"prompts/{args.suite}.json")))["prompts"]
    pids = [p["id"] for p in prompts]
    mdir = os.path.join(ROOT, "artifacts", "metrics")
    os.makedirs(mdir, exist_ok=True)

    rows, pertok_rows = [], []
    for pair in args.pairs:
        a, b = pair.split(":")
        if not os.path.isdir(os.path.join(ROOT, "artifacts", "tensors", a)):
            print(f"[skip] arm {a} missing"); continue
        for pid in pids:
            if not os.path.exists(os.path.join(ROOT, "artifacts", "tensors", a, f"{pid}.npz")):
                print(f"[skip] {a}/{pid} missing"); continue
            if not os.path.exists(os.path.join(ROOT, "artifacts", "tensors", b, f"{pid}.npz")):
                print(f"[skip] {b}/{pid} missing"); continue
            row = compare_pair(a, b, pid)
            rows.append(row)
            pe_a, _ = load_arm(a, pid, "prompt_embeds")
            pe_b, _ = load_arm(b, pid, "prompt_embeds")
            c = per_token_cosine(pe_a, pe_b)
            for i, ci in enumerate(c):
                pertok_rows.append({"pair": row["pair"], "pid": pid, "token_index": i, "cosine": float(ci)})

    if not rows:
        print("nothing to compare"); return
    keys = sorted({k for r in rows for k in r if k not in ("pair", "pid")})
    suffix = "_".join(a+b for a,b in [x.split(":") for x in args.pairs])
    with open(os.path.join(mdir, f"embedding_summary_{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "pid"] + keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(mdir, f"embedding_per_token_{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "pid", "token_index", "cosine"])
        w.writeheader()
        for r in pertok_rows:
            w.writerow(r)

    # aggregate stats per pair
    agg = {}
    for r in rows:
        agg.setdefault(r["pair"], []).append(r)
    summary = {}
    for pair, rs in agg.items():
        summary[pair] = {
            "prenorm.cosine": stats_block([r["prenorm.cosine"] for r in rs]),
            "prenorm.rel_l2": stats_block([r["prenorm.rel_l2"] for r in rs]),
            "prenorm.max_abs": stats_block([r["prenorm.max_abs"] for r in rs]),
            "prenorm.bf16_exact": stats_block([r["prenorm.bf16_exact_match"] for r in rs]),
            "prompt_embeds.cosine": stats_block([r["prompt_embeds.cosine"] for r in rs]),
            "prompt_embeds.rel_l2": stats_block([r["prompt_embeds.rel_l2"] for r in rs]),
            "prompt_embeds.bf16_exact": stats_block([r["prompt_embeds.bf16_exact_match"] for r in rs]),
            "embeds.cosine": stats_block([r["embeds.cosine"] for r in rs]),
        }
    with open(os.path.join(mdir, f"embedding_summary_stats_{suffix}_{args.suite}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: {"prenorm.cos_median": v["prenorm.cosine"]["median"],
                          "prenorm.rel_l2_median": v["prenorm.rel_l2"]["median"],
                          "bf16_exact_median": v["prenorm.bf16_exact"]["median"]}
                      for k, v in summary.items()}, indent=2))

    # plots
    pdir = os.path.join(ROOT, "artifacts", "plots")
    os.makedirs(pdir, exist_ok=True)
    pairs = list(summary.keys())
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pairs)), 5))
    data = [[r["prenorm.rel_l2"] for r in agg[p]] for p in pairs]
    bp = ax.boxplot(data, tick_labels=pairs, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 (prenorm, full seq)")
    ax.set_title("Prompt embedding error distribution by arm pair")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "embedding_histogram.png"), dpi=140)

    # per-prompt cosine heatmap (pairs x prompts)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(pids)), 0.8 * len(pairs) + 2))
    M = np.full((len(pairs), len(pids)), np.nan)
    for i, p in enumerate(pairs):
        for j, pid in enumerate(pids):
            rs = [r for r in agg[p] if r["pid"] == pid]
            if rs:
                M[i, j] = rs[0]["prenorm.cosine"]
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=min(0.99, np.nanmin(M) if not np.all(np.isnan(M)) else 0.99))
    ax.set_yticks(range(len(pairs)), pairs)
    ax.set_xticks(range(len(pids)), pids, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, label="cosine (prenorm)")
    ax.set_title("Embedding error heatmap (pair x prompt)")
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "embedding_error_heatmap.png"), dpi=140)
    print("plots saved")


if __name__ == "__main__":
    main()
