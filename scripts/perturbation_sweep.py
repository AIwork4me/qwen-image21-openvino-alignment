#!/usr/bin/env python3
"""Phase H: embedding perturbation sensitivity calibration.

delta = E_OV - E_ROCM (per prompt). Construct scaled/2x variants and norm-matched random
perturbations, run the SAME diffusion setup for 25 and 40 steps, measure final latent and
image distances. Determines whether small conditioning deltas naturally produce large
final-image differences (chaotic sensitivity) independent of OpenVINO.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, tensor_hash, paired_metrics

from safetensors.torch import load_file


def load_pipe(cfg):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(cfg["model"]["snapshot_path"], torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    pipe.processor = None
    pipe.to("cuda")
    return pipe


def gen(pipe, lat_cpu, embeds_fp32, steps, resolution):
    if embeds_fp32.ndim == 2:
        embeds_fp32 = embeds_fp32.unsqueeze(0)
    pe = embeds_fp32.to("cuda", dtype=torch.bfloat16)
    lat = lat_cpu.to("cuda", dtype=torch.bfloat16).clone()
    latents_trace = []

    def cb(p, i, t, cbk):
        latents_trace.append(cbk["latents"].detach().float().cpu().numpy())
        return {}

    out = pipe(prompt=None, prompt_embeds=pe, prompt_embeds_mask=None,
               num_inference_steps=steps, height=resolution, width=resolution,
               latents=lat, true_cfg_scale=1.0, negative_prompt=None,
               callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
               output_type="pil")
    return out.images[0], latents_trace


def norm_matched_random_delta(ref, rng):
    """Gaussian perturbation with per-token norm matching of the reference delta."""
    d = rng.standard_normal(ref.shape).astype(np.float32)
    tok_norms_ref = np.linalg.norm(ref, axis=1, keepdims=True)  # [seq,1]
    tok_norms_d = np.linalg.norm(d, axis=1, keepdims=True)
    d = d * (tok_norms_ref / np.maximum(tok_norms_d, 1e-12))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--pid", default="S01")
    ap.add_argument("--arm-ref", default="R0")
    ap.add_argument("--arm-ov", default="O3")
    ap.add_argument("--resolutions-steps", default="1024:25,1024:40")
    ap.add_argument("--random-seeds", default="11,22,33")
    ap.add_argument("--scales", default="0.25,0.5,1.0,2.0")
    args = ap.parse_args()

    cfg = load_cfg()
    pipe = load_pipe(cfg)
    za = np.load(os.path.join(ROOT, "artifacts", "tensors", args.arm_ref, f"{args.pid}.npz"))
    zb = np.load(os.path.join(ROOT, "artifacts", "tensors", args.arm_ov, f"{args.pid}.npz"))
    E0 = za["prompt_embeds_bf16_view"].astype(np.float32)
    E1 = zb["prompt_embeds_bf16_view"].astype(np.float32)
    delta = E1 - E0
    d_norm = float(np.linalg.norm(delta))
    base_metrics = paired_metrics(E0, E1)

    variants = {"E0": E0}
    for s in [float(x) for x in args.scales.split(",")]:
        variants[f"E{s:g}xOVdelta"] = E0 + s * delta
    for sd in [int(x) for x in args.random_seeds.split(",")]:
        rng = np.random.default_rng(sd)
        variants[f"R{sd}_normmatched"] = E0 + norm_matched_random_delta(delta, rng)

    results = {"pid": args.pid, "delta_norm": d_norm, "delta_rel_l2": base_metrics["rel_l2"],
               "delta_cosine": base_metrics["cosine"], "runs": []}
    imgdir = os.path.join(ROOT, "artifacts", "images")
    os.makedirs(imgdir, exist_ok=True)

    for rs in args.resolutions_steps.split(","):
        res, steps = (int(x) for x in rs.split(":"))
        lat = load_file(os.path.join(ROOT, "artifacts", "latents", f"initial_latent_{res}.safetensors"))["latent"]
        ref_lat = None
        for name, E in variants.items():
            img, trace = gen(pipe, lat, torch.from_numpy(E.astype(np.float32)), steps, res)
            fn = f"pert_{args.pid}_{res}_{steps}s_{name}.png"
            img.save(os.path.join(imgdir, fn))
            if name == "E0":
                ref_lat = trace[-1]
                ref_img = img
            else:
                lm = paired_metrics(ref_lat, trace[-1])
                from PIL import Image
                a = np.asarray(ref_img, dtype=np.float32)
                b = np.asarray(img, dtype=np.float32)
                mse = float(((a - b) ** 2).mean())
                psnr = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
                results["runs"].append({"res": res, "steps": steps, "variant": name,
                                        "latent.cosine": lm["cosine"], "latent.rel_l2": lm["rel_l2"],
                                        "psnr": psnr})
                print(f"[{res}/{steps}][{name}] latent cos {lm['cosine']:.5f} rel_l2 {lm['rel_l2']:.5f} psnr {psnr:.2f}")

    save_json(results, os.path.join(ROOT, "artifacts", "metrics", f"perturbation_{args.pid}.json"))
    print("saved perturbation results")


if __name__ == "__main__":
    main()
