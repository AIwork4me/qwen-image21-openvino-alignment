#!/usr/bin/env python3
"""Generate the required key figures (Section 32) from raw metric artifacts."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.join(ROOT, "artifacts", "plots")
os.makedirs(P, exist_ok=True)


def fig_trace_curves():
    for steps in (25, 40):
        tn = "trace_S02_25s.json" if steps == 25 else "trace_S02_40s_O3.json"
        t = load_json(os.path.join(ROOT, "artifacts", "metrics", tn))
        ss = [s for s in t["step_stats"]]
        xs = [s["step"] for s in ss]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(xs, [1 - s["latent.cosine"] for s in ss], "o-", ms=3)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("diffusion step"); axes[0].set_ylabel("1 - latent cosine")
        axes[0].set_title(f"{steps}-step latent divergence (R0 vs OV-INT8)"); axes[0].grid(alpha=0.3)
        axes[1].plot(xs, [s["latent.rel_l2"] for s in ss], "o-", ms=3, color="tab:red")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("diffusion step"); axes[1].set_ylabel("latent rel L2")
        axes[1].set_title(f"{steps}-step latent rel L2"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(P, f"diffusion_latent_divergence_{steps}steps.png"), dpi=140)
        plt.close(fig)
        # model output divergence
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [s["model_out.rel_l2"] for s in ss], "s-", ms=3, color="tab:purple", label="model output rel L2")
        ax.plot(xs, [s["latent.rel_l2"] for s in ss], "o-", ms=3, color="tab:red", label="latent rel L2")
        ax.set_yscale("log"); ax.set_xlabel("diffusion step"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"transformer output vs latent divergence ({steps} steps)")
        fig.tight_layout(); fig.savefig(os.path.join(P, f"diffusion_model_output_{steps}steps.png"), dpi=140)
        plt.close(fig)


def fig_step_compare():
    t25 = load_json(os.path.join(ROOT, "artifacts", "metrics", "trace_S02_25s.json"))["step_stats"]
    t40 = load_json(os.path.join(ROOT, "artifacts", "metrics", "trace_S02_40s_O3.json"))["step_stats"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([s["step"] for s in t25], [s["latent.rel_l2"] for s in t25], "o-", ms=3, label="25-step schedule")
    ax.plot([s["step"] for s in t40], [s["latent.rel_l2"] for s in t40], "s-", ms=3, label="40-step schedule")
    ax.set_yscale("log"); ax.set_xlabel("diffusion step"); ax.set_ylabel("latent rel L2 (R0 vs OV-INT8)")
    ax.set_title("25 vs 40 step trajectory divergence (same embeddings delta, same latent)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(P, "steps_25_vs_40_comparison.png"), dpi=140)
    plt.close(fig)


def fig_perturbation():
    d = load_json(os.path.join(ROOT, "artifacts", "metrics", "perturbation_S02.json"))
    rows = d["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel, fname in [(axes[0], "latent.rel_l2", "final latent rel L2", "rel_l2"),
                                   (axes[1], "psnr", "final image PSNR (dB)", "psnr")]:
        for steps, mk in ((25, "o"), (40, "s")):
            rs = [r for r in rows if r["steps"] == steps]
            names = [r["variant"] for r in rs]
            vals = [r[key] for r in rs]
            ax.plot(range(len(vals)), vals, mk + "-", label=f"{steps} steps")
            ax.set_xticks(range(len(vals)), names, rotation=60, fontsize=6)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend()
    axes[0].axhline(d["delta_rel_l2"], color="k", ls="--", lw=1, label="embedding delta magnitude")
    fig.suptitle("Perturbation magnitude vs final divergence: OV delta vs norm-matched random deltas (S02)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "perturbation_vs_final_divergence.png"), dpi=140)
    plt.close(fig)


def fig_precision_chain():
    pairs = [
        ("ROCm repeat", 0.0), ("OV FP32 conv.", 7.7e-6), ("FP16 storage", 5.7e-5),
        ("OV BF16 runtime", 1.28e-2), ("GPU BF16 vs FP32", 5.28e-2),
        ("OV INT8 vs OV FP32", 6.96e-2), ("GPU BF16 vs OV INT8", 9.67e-2),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    ax.bar(range(len(vals)), vals, color=["tab:green" if v < 0.02 else ("tab:orange" if v < 0.08 else "tab:red") for v in vals])
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("conditioning rel L2 (median)")
    ax.set_title("Error chain: where does divergence come from? (n=100 for INT8 pairs)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(P, "precision_chain.png"), dpi=140)
    plt.close(fig)


def fig_quality_tradeoff():
    perf = load_json(os.path.join(ROOT, "artifacts", "metrics", "cpu_performance.json"))
    arms, lat, load, err = [], [], [], []
    errmap = {"O0": 7.7e-6, "O1": 1.28e-2, "O3r": 6.96e-2}
    for a in perf:
        arm = a.get("arm")
        if arm not in errmap:
            continue
        arms.append(arm)
        lat.append(a["prompts"]["P_long"]["warm_median_s"])
        load.append(a.get("load_s", 0))
        err.append(errmap[arm])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    ax1.bar([a + " load" for a in arms], load, alpha=0.5, color="tab:blue", label="model load s")
    ax1.bar([a + " warm@171tok" for a in arms], lat, color="tab:cyan", label="warm encode s (171 tok)")
    ax2.plot([a + " load" for a in arms] + [a + " warm@171tok" for a in arms],
             err * 2, "ro", label="embedding rel L2")
    ax2.set_yscale("log"); ax2.set_ylabel("embedding rel L2 (vs OV FP32 / GPU FP32)", color="r")
    ax1.set_ylabel("seconds")
    ax1.set_title("OpenVINO CPU: accuracy vs performance tradeoff (EPYC 9334)")
    fig.tight_layout(); fig.savefig(os.path.join(P, "ov_accuracy_performance_tradeoff.png"), dpi=140)
    plt.close(fig)


def fig_100suite_distributions():
    import csv
    f = "artifacts/metrics/image_metrics_alignment_100___s20261001_40s_R0_vs_O3r.csv"
    rows = list(csv.DictReader(open(f)))
    lp = [float(r["lpips"]) for r in rows if r.get("lpips")]
    ps = [float(r["psnr"]) for r in rows if r.get("psnr")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(lp, bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("LPIPS (R0 vs OV-INT8, 40 steps)"); axes[0].set_ylabel("# prompts")
    axes[0].set_title(f"Perceptual difference distribution, n={len(lp)} (median {np.median(lp):.3f})")
    axes[1].hist(ps, bins=30, color="tab:green", alpha=0.8)
    axes[1].set_xlabel("PSNR (dB)")
    axes[1].set_title(f"Pixel difference distribution (median {np.median(ps):.1f} dB)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(P, "suite100_quality_distributions.png"), dpi=140)
    plt.close(fig)


def fig_text_rendering():
    import csv
    f = "artifacts/metrics/image_metrics_alignment_100___s20261001_40s_R0_vs_O3r.csv"
    rows = [r for r in csv.DictReader(open(f)) if r.get("cer_a")]
    if not rows:
        return
    ca = [float(r["cer_a"]) for r in rows]
    cb = [float(r["cer_b"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot([ca, cb], tick_labels=[f"GPU reference (R0)", "OpenVINO INT8 (O3r)"], showfliers=True)
    ax.set_ylabel("OCR character error rate (40 steps, 20 text prompts)")
    ax.set_title("Text-rendering accuracy: no regression (paired median diff = 0.0)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(P, "text_rendering_accuracy.png"), dpi=140)
    plt.close(fig)


def fig_arm_tail_comparison():
    """LPIPS distributions by arm (canary 40s + 2K spot): BF16 tightens the tail."""
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    data, labels = [], []
    for arm in ("O3r", "O3qL", "O1"):
        f = f"artifacts/metrics/image_metrics_canary___s20261001_40s_R0_vs_{arm}.csv"
        if os.path.exists(f):
            rows = list(csv.DictReader(open(f)))
            data.append([float(r["lpips"]) for r in rows if r.get("lpips")])
            labels.append(f"INT8 ({arm})" if arm != "O1" else "BF16 (O1)")
    axes[0].boxplot(data, tick_labels=labels)
    axes[0].set_ylabel("LPIPS vs GPU reference")
    axes[0].set_title("Canary 40-step (n=10): BF16 tightens the worst case")
    axes[0].grid(alpha=0.3, axis="y")
    d2 = []
    for arm in ("O3r", "O1"):
        f = f"artifacts/metrics/image_metrics_production_30_P*_s20261001_40s_R0_vs_{arm}.csv"
        fs = glob.glob(f)
        if fs:
            rows = list(csv.DictReader(open(fs[0])))
            d2.append([float(r["lpips"]) for r in rows if r.get("lpips")])
    if d2:
        axes[1].boxplot(d2, tick_labels=["INT8 (O3r)", "BF16 (O1)"])
        axes[1].set_title("2K production spot (n=4): same direction")
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "arm_tail_comparison.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_arm_tail_comparison()
    fig_trace_curves()
    fig_step_compare()
    fig_perturbation()
    fig_precision_chain()
    fig_quality_tradeoff()
    fig_100suite_distributions()
    fig_text_rendering()
    print("key figures saved to", P)
    print("\n".join(sorted(os.listdir(P))))
