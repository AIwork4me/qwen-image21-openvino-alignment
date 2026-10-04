#!/usr/bin/env python3
"""Phase J suite driver: encode -> paired diffusion (25 & 40 steps) -> images + traces,
loading the pipeline ONCE. Arms compared against R0 under identical frozen inputs.

Usage:
  python scripts/benchmark_suite.py --suite canary --arms R0,O3 --steps 25,40 --seeds 20261001
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, tensor_hash, paired_metrics

from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--arms", default="R0,O3")
    ap.add_argument("--steps", default="25,40")
    ap.add_argument("--seeds", default="20261001")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--resource-csv", default=None)
    ap.add_argument("--pids", default=None, help="comma list to restrict to a prompt subset")
    args = ap.parse_args()

    cfg = load_cfg()
    arms = args.arms.split(",")
    steps_list = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    prompts = json.load(open(os.path.join(ROOT, f"prompts/{args.suite}.json")))["prompts"]
    if args.pids:
        keep = set(args.pids.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    for p in prompts:
        for arm in arms:
            f = os.path.join(ROOT, "artifacts", "tensors", arm, f"{p['id']}.npz")
            assert os.path.exists(f), f"missing embeddings for {arm}/{p['id']} — run encoders first"

    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(cfg["model"]["snapshot_path"], torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    pipe.processor = None
    pipe.to("cuda")
    if args.resolution >= 2048:
        # RECORDED CHANGE (Section 38): tiled VAE decode required to fit 2K in VRAM.
        # Applied identically to BOTH arms; decode math unchanged per tile.
        pipe.vae.enable_tiling()

    lat_base = load_file(os.path.join(ROOT, "artifacts", "latents", f"initial_latent_{args.resolution}.safetensors"))["latent"]

    imgdir = os.path.join(ROOT, "artifacts", "images")
    os.makedirs(imgdir, exist_ok=True)
    monitor = None
    if args.resource_csv:
        from common import ResourceMonitor
        monitor = ResourceMonitor(args.resource_csv, hz=cfg["resource_monitor"]["sample_hz"]).start()

    def gen(embeds, steps, lat):
        def cb(pp, i, t, cbk):
            return {}
        return pipe(prompt=None,
                    prompt_embeds=embeds.to("cuda", dtype=torch.bfloat16),
                    prompt_embeds_mask=None,
                    num_inference_steps=steps,
                    height=args.resolution, width=args.resolution,
                    latents=lat.to("cuda", dtype=torch.bfloat16).clone(),
                    true_cfg_scale=1.0, negative_prompt=None,
                    callback_on_step_end=cb, output_type="pil").images[0]

    manifest_rows = []
    t_start = time.time()
    for seed in seeds:
        g = torch.Generator("cpu").manual_seed(seed)
        lat = lat_base.clone()
        # deterministic per-(seed) perturbation of the frozen base latent to vary seeds while
        # staying CPU-reproducible: lat_seed = base + eps is NOT used; instead regenerate fully:
        h = w = args.resolution // 16
        lat = torch.randn(1, 1, 64, h, w, generator=g, dtype=torch.float32)
        lat = lat.view(1, 64, h * w).transpose(1, 2).contiguous()
        lat_hash = tensor_hash(lat.numpy())
        for p in prompts:
            pid = p["id"]
            imgs = {}
            for arm in arms:
                z = np.load(os.path.join(ROOT, "artifacts", "tensors", arm, f"{pid}.npz"))
                pe = torch.from_numpy(z["prompt_embeds_bf16_view"].astype(np.float32)).unsqueeze(0)
                for steps in steps_list:
                    fn = f"{args.suite}_{pid}_s{seed}_{steps}s_{arm}.png"
                    if os.path.exists(os.path.join(imgdir, fn)):
                        continue
                    torch.cuda.synchronize(); t0 = time.time()
                    img = gen(pe, steps, lat)
                    torch.cuda.synchronize()
                    gen_s = time.time() - t0
                    img.save(os.path.join(imgdir, fn))
                    manifest_rows.append({"suite": args.suite, "pid": pid, "seed": seed, "steps": steps,
                                          "arm": arm, "latent_hash": lat_hash, "gen_s": gen_s,
                                          "pe_hash": tensor_hash(pe.numpy()), "file": fn})
                    print(f"[{pid} s{seed} {steps}s {arm}] {gen_s:.1f}s")
    if monitor:
        monitor.stop()
    save_json({"rows": manifest_rows, "total_s": time.time() - t_start},
              os.path.join(ROOT, "artifacts", "metrics", f"suite_{args.suite}_manifest.json"))
    print("suite complete", args.suite, len(manifest_rows), "gens in", time.time() - t_start, "s")


if __name__ == "__main__":
    main()
