#!/usr/bin/env python3
"""Phase G: 25->40 crossover / causal localization on the 40-step schedule.

Manual replication of the QwenImage21 denoising loop (use_kv_cache=False for all arms
so embedding swaps mid-run are sound; math identical, cache is an optimization).

Design (k = crossover step, default 25):
  A: latent_k(R0-embeds)    + R0 embeds  -> continue to 40   [pure reference continuation]
  B: latent_k(R0-embeds)    + O3 embeds  -> continue to 40   [swap conditioning]
  C: latent_k(O3-embeds)    + R0 embeds  -> continue to 40   [swap latent state]
  D: latent_k(O3-embeds)    + O3 embeds  -> continue to 40   [pure OV continuation]
Distinguishes accumulated latent-state divergence vs ongoing conditioning sensitivity.
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
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps



def make_schedule(snap_cfg, resolution, steps):
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(snap_cfg)
    seq_len = (resolution // 16) ** 2
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(seq_len, sched.config.get("base_image_seq_len", 256),
                         sched.config.get("max_image_seq_len", 4096),
                         sched.config.get("base_shift", 0.5),
                         sched.config.get("max_shift", 1.15))
    s = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
    timesteps, _ = retrieve_timesteps(s, steps, "cuda", sigmas=sigmas, mu=mu)
    return s, timesteps


def run_steps(pipe, lat, embeds, timesteps, sched, from_i, to_i, img_shapes, img_mask, resolution, trace=None):
    lat = lat.clone()
    with torch.no_grad():
        for i in range(from_i, to_i):
            t = timesteps[i]
            timestep = t.expand(lat.shape[0]).to(lat.dtype)
            noise_pred = pipe.transformer(
                hidden_states=lat,
                timestep=timestep / 1000,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=None,
                img_shapes=img_shapes,
                img_mask=img_mask,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, -lat.size(1):]
            lat = sched.step(noise_pred, t, lat, return_dict=False)[0].to(lat.device)
            if trace is not None:
                trace.append(lat.detach().float().cpu().numpy())
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--pid", default="S01")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=25, help="crossover step index (0-based)")
    ap.add_argument("--arm-a", default="R0")
    ap.add_argument("--arm-b", default="O3")
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    pipe = load_pipe_min(cfg)
    lat0 = load_file(os.path.join(ROOT, "artifacts", "latents", f"initial_latent_{args.resolution}.safetensors"))["latent"].to("cuda", dtype=torch.bfloat16)

    def emb(arm):
        z = np.load(os.path.join(ROOT, "artifacts", "tensors", arm, f"{args.pid}.npz"))
        e = torch.from_numpy(z["prompt_embeds_bf16_view"].astype(np.float32))
        if e.ndim == 2:
            e = e.unsqueeze(0)
        return e.to("cuda", dtype=torch.bfloat16)

    ea, eb = emb(args.arm_a), emb(args.arm_b)
    seq = (args.resolution // 16) ** 2
    img_shapes = [[(1, args.resolution // 16, args.resolution // 16)]]
    text_len = ea.shape[1]
    # joint mask: text slots (0) + one slot per 2x2 group of target latents (1) — mirrors append_target_slots
    img_mask = torch.cat([torch.zeros(1, text_len, dtype=torch.bool, device="cuda"),
                          torch.ones(1, seq // 4, dtype=torch.bool, device="cuda")], dim=1)

    k = args.k
    # Phase 1: run both arms to step k
    lat_at_k = {}
    for arm, e in ((args.arm_a, ea), (args.arm_b, eb)):
        sched, timesteps = make_schedule(os.path.join(snap, "scheduler"), args.resolution, args.steps)
        sched.set_begin_index(0)
        lat_at_k[arm] = run_steps(pipe, lat0, e, timesteps, sched, 0, k, img_shapes, img_mask, args.resolution)
        print(f"[{arm}] ran to step {k}")

    # Phase 2: four continuations
    branches = {
        "A_ref": (args.arm_a, lat_at_k[args.arm_a]),
        "B_swap_cond": (args.arm_b, lat_at_k[args.arm_a]),
        "C_swap_latent": (args.arm_a, lat_at_k[args.arm_b]),
        "D_ov": (args.arm_b, lat_at_k[args.arm_b]),
    }
    finals, final_lats = {}, {}
    for name, (arm, latk) in branches.items():
        sched, timesteps = make_schedule(os.path.join(snap, "scheduler"), args.resolution, args.steps)
        # advance scheduler internal state to step k by replaying step indices
        sched.set_begin_index(k)
        latf = run_steps(pipe, latk, emb(arm), timesteps, sched, k, args.steps, img_shapes, img_mask, args.resolution)
        finals[name] = latf
        final_lats[name] = latf.float().cpu().numpy()
        print(f"[{name}] continued {k}->{args.steps}")

    # decode images (replicating the pipeline's VAE postprocessing exactly)
    imgdir = os.path.join(ROOT, "artifacts", "images")
    os.makedirs(imgdir, exist_ok=True)
    for name, latf in finals.items():
        lat = latf.transpose(1, 2).reshape(1, 64, 1, args.resolution // 16, args.resolution // 16).to(pipe.vae.dtype)
        lm = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        ls = torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        with torch.no_grad():
            img = pipe.vae.decode(lat * ls + lm, return_dict=False)[0][:, :, 0]
        pi = pipe.image_processor.postprocess(img, output_type="pil")[0]
        pi.save(os.path.join(imgdir, f"crossover_{args.pid}_{args.steps}s_k{k}_{name}.png"))

    out = {"pid": args.pid, "steps": args.steps, "k": k, "arms": [args.arm_a, args.arm_b],
           "pairs": {}}
    for n1 in finals:
        for n2 in finals:
            if n1 < n2:
                m = paired_metrics(final_lats[n1], final_lats[n2])
                out["pairs"][f"{n1}|{n2}"] = {"cosine": m["cosine"], "rel_l2": m["rel_l2"]}
    save_json(out, os.path.join(ROOT, "artifacts", "metrics", f"crossover_{args.pid}_k{k}.json"))
    print(json.dumps(out["pairs"], indent=2))


def load_pipe_min(cfg):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(cfg["model"]["snapshot_path"], torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    pipe.processor = None
    pipe.to("cuda")
    return pipe


if __name__ == "__main__":
    main()
