#!/usr/bin/env python3
"""v2 population transplant suite: embeddings from {C_BF16,C_INT8,C_W4A8[,+O arms]}
through the identical frozen-latent diffusion backend at 25 and 40 steps.

Per prompt and step count, saves images + final latents and computes paired
divergence vs the C_BF16 reference:
  D  = rel_l2(final latent INT8 vs BF16), rel_l2(W4A8 vs BF16)
  image PSNR/SSIM (cheap, in-process)

Outputs:
  /valdata/images/population/{pid}_{steps}s_{arm}.png
  artifacts/v2/metrics/population_runs.csv        (one row per generation)
  artifacts/v2/metrics/population_pairs.csv       (paired D25/D40 per prompt)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C
import v2_gen as G


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean()
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=255))


def to_gray(np_img: np.ndarray) -> np.ndarray:
    if np_img.ndim == 3:
        return (0.299 * np_img[..., 0] + 0.587 * np_img[..., 1] + 0.114 * np_img[..., 2]).astype(np.uint8)
    return np_img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="alignment_100")
    ap.add_argument("--arms", default="C_BF16,C_INT8,C_W4A8")
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 40])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N prompts")
    ap.add_argument("--track-resources", action="store_true")
    ap.add_argument("--suffix", default="", help="embedding npz suffix for seed variants")
    ap.add_argument("--out-tag", default="", help="output csv tag (e.g. _seeds)")
    ap.add_argument("--latent-suffix", default="", help="latent file suffix, e.g. _s20261002")
    args = ap.parse_args()

    C.gate_environment()
    prompts = C.load_prompts(args.suite)
    if args.limit:
        prompts = prompts[: args.limit]
    arms = args.arms.split(",")
    ref_arm = "C_BF16"
    imgdir = os.path.join(C.VALDATA, "images", "population")
    os.makedirs(imgdir, exist_ok=True)
    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    os.makedirs(mdir, exist_ok=True)
    runs_csv = os.path.join(mdir, f"population_runs{args.out_tag}.csv")
    pairs_csv = os.path.join(mdir, f"population_pairs{args.out_tag}.csv")

    done = set()
    if os.path.exists(runs_csv):
        with open(runs_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["pid"], int(row["steps"]), row["arm"]))
        print(f"resuming: {len(done)} generations already recorded")

    latent = G.get_latent(args.resolution, args.latent_suffix or None)
    latent_sha = C.tensor_sha256(latent)

    runs_fieldnames = ["pid", "category", "steps", "arm", "gen_s", "latent_sha256",
                       "pe_sha256", "final_latent_rel_l2_vs_ref", "psnr_vs_ref", "ssim_vs_ref",
                       "vram_peak_mb", "rss_peak_mb"]
    final_lats: dict[tuple, np.ndarray] = {}
    imgs: dict[tuple, np.ndarray] = {}

    mon = {"stop": False, "vram": 0, "rss": 0}
    if args.track_resources:
        import threading

        import psutil

        proc = psutil.Process()

        def sampler():
            while not mon["stop"]:
                try:
                    mon["vram"] = max(mon["vram"], torch.cuda.memory_allocated() // (1024 * 1024))
                    mon["rss"] = max(mon["rss"], proc.memory_info().rss // (1024 * 1024))
                except Exception:
                    pass
                time.sleep(0.2)
        threading.Thread(target=sampler, daemon=True).start()

    newf = not os.path.exists(runs_csv)
    with open(runs_csv, "a", newline="") as fruns:
        wruns = csv.DictWriter(fruns, fieldnames=runs_fieldnames)
        if newf:
            wruns.writeheader()
        for pid_data in prompts:
            pid = pid_data["id"]
            for steps in args.steps:
                for arm in arms:
                    key = (pid, steps, arm)
                    if key in done:
                        # reload cached outputs for pairing
                        p = os.path.join(imgdir, f"{pid}_{steps}s_{arm}.png")
                        fl = os.path.join(imgdir, f"{pid}_{steps}s_{arm}_flat.npy")
                        if os.path.exists(p) and os.path.exists(fl):
                            from PIL import Image
                            imgs[key] = np.array(Image.open(p))
                            final_lats[key] = np.load(fl)
                        continue
                    pe = G.get_embeds(arm, pid)
                    if args.suffix:
                        pe = torch.from_numpy(
                            np.load(os.path.join(C.VALDATA, "tensors", arm, f"{pid}{args.suffix}.npz"),
                                    allow_pickle=True)["cond"].astype(np.float32))
                        if pe.ndim == 2:
                            pe = pe.unsqueeze(0)
                    mon["vram"] = mon["rss"] = 0
                    img, flat, _, dt = G.controlled_run(pe, steps, args.resolution)
                    tag = f"{args.suffix}" if args.suffix else ""
                    img.save(os.path.join(imgdir, f"{pid}{tag}_{steps}s_{arm}.png"))
                    np.save(os.path.join(imgdir, f"{pid}{tag}_{steps}s_{arm}_flat.npy"), flat)
                    imgs[key] = np.array(img)
                    final_lats[key] = flat
                    row = {"pid": pid, "category": pid_data.get("category", ""),
                           "steps": steps, "arm": arm, "gen_s": round(dt, 2),
                           "latent_sha256": latent_sha,
                           "pe_sha256": C.tensor_sha256(pe), "final_latent_rel_l2_vs_ref": "",
                           "psnr_vs_ref": "", "ssim_vs_ref": "",
                           "vram_peak_mb": mon["vram"] if args.track_resources else "",
                           "rss_peak_mb": mon["rss"] if args.track_resources else ""}
                    if arm == ref_arm:
                        wruns.writerow(row)
                        fruns.flush()
                    else:
                        rk = (pid, steps, ref_arm)
                        if rk in final_lats:
                            row["final_latent_rel_l2_vs_ref"] = C.rel_l2(
                                torch.from_numpy(final_lats[rk]), torch.from_numpy(flat))
                            row["psnr_vs_ref"] = psnr(imgs[rk], imgs[key])
                            row["ssim_vs_ref"] = ssim_gray(to_gray(imgs[rk]), to_gray(imgs[key]))
                        wruns.writerow(row)
                        fruns.flush()
                    print(f"[{pid} {steps}s {arm}] {dt:.1f}s", flush=True)

    # paired population table
    pair_rows = []
    for pid_data in prompts:
        pid = pid_data["id"]
        for steps in args.steps:
            base = None
            with open(runs_csv) as f:
                for row in csv.DictReader(f):
                    if row["pid"] == pid and int(row["steps"]) == steps and row["arm"] == ref_arm:
                        base = row
            if base is None:
                continue
            with open(runs_csv) as f:
                for row in csv.DictReader(f):
                    if row["pid"] == pid and int(row["steps"]) == steps and row["arm"] != ref_arm \
                            and row["arm"] in arms:
                        pair_rows.append({
                            "pid": pid, "category": pid_data.get("category", ""),
                            "steps": steps, "vs": row["arm"],
                            "latent_rel_l2": row["final_latent_rel_l2_vs_ref"],
                            "psnr": row["psnr_vs_ref"], "ssim": row["ssim_vs_ref"]})
    with open(pairs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "category", "steps", "vs",
                                          "latent_rel_l2", "psnr", "ssim"])
        w.writeheader()
        w.writerows(pair_rows)

    C.save_json({
        "experiment_id": "V2-POPULATION",
        "suite": args.suite, "arms": arms, "steps": args.steps,
        "resolution": args.resolution, "n_prompts": len(prompts),
        "controls": {"latent_sha256": latent_sha,
                     "true_cfg": 1.0, "negative": None,
                     "schedule": "FlowMatchEulerDiscreteScheduler (frozen manifest)"},
        "n_generations": len(done) + len(final_lats),
    }, os.path.join(mdir, f"population_manifest{args.out_tag}.json"))
    print("pairs ->", pairs_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
