#!/usr/bin/env python3
"""Generate frozen initial latents (PACKED [B, seq, C]) and record exact 25/40-step schedules.

Replicates QwenImage21Pipeline.__call__ schedule construction:
  sigmas = linspace(1.0, 1/steps, steps)
  mu = calculate_shift(seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift)
  timesteps = retrieve_timesteps(scheduler, steps, sigmas=sigmas, mu=mu)
Latents: randn on CPU (device-independent), shape (1, 1, C, H/16, W/16) packed to (1, seq, C).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, tensor_hash

from safetensors.torch import save_file
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps



def main():
    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    seed = cfg["latent_seed"]
    ldir = os.path.join(ROOT, "artifacts", "latents")
    os.makedirs(ldir, exist_ok=True)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
    lat_channels = 64

    schedules = {}
    for res in (1024, 2048):
        seq_len = (res // 16) * (res // 16)
        for steps in (25, 40):
            sigmas = np.linspace(1.0, 1 / steps, steps)
            mu = calculate_shift(seq_len,
                                 sched.config.get("base_image_seq_len", 256),
                                 sched.config.get("max_image_seq_len", 4096),
                                 sched.config.get("base_shift", 0.5),
                                 sched.config.get("max_shift", 1.15))
            s2 = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
            timesteps, n = retrieve_timesteps(s2, steps, "cpu", sigmas=sigmas, mu=mu)
            schedules[f"{res}_{steps}"] = {
                "seq_len": seq_len, "mu": float(mu),
                "timesteps": timesteps.tolist(),
                "sigmas": s2.sigmas.tolist(),
                "scheduler_config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                     for k, v in sched.config.items()
                                     if isinstance(v, (int, float, bool, str, list, tuple, type(None)))},
            }

    for res in (1024, 2048):
        g = torch.Generator("cpu").manual_seed(seed)
        h = w = res // 16
        raw = torch.randn(1, 1, lat_channels, h, w, generator=g, dtype=torch.float32)
        packed = raw.view(1, lat_channels, h * w).transpose(1, 2).contiguous()  # [1, seq, 64]
        save_file({"latent": packed}, os.path.join(ldir, f"initial_latent_{res}.safetensors"),
                  metadata={"seed": str(seed), "resolution": str(res), "layout": "packed [1,seq,64]"})
        print(f"latent {res}: packed {tuple(packed.shape)} hash {tensor_hash(packed.numpy())}")

    save_json({"seed": seed, "schedules": schedules},
              os.path.join(ldir, "latent_and_schedule_manifest.json"))

    chk = {}
    for res in (1024, 2048):
        t25 = np.array(schedules[f"{res}_25"]["timesteps"])
        t40 = np.array(schedules[f"{res}_40"]["timesteps"])
        s25 = np.array(schedules[f"{res}_25"]["sigmas"])
        s40 = np.array(schedules[f"{res}_40"]["sigmas"])
        chk[f"{res}_timesteps_25_eq_first25_of_40"] = bool(np.allclose(t25, t40[:25])) and len(t25) == 25
        chk[f"{res}_sigma_25_eq_first25_of_40"] = bool(np.allclose(s25[:25], s40[:25])) and len(s25) == 26
        chk[f"{res}_n_timesteps"] = {"25": len(t25), "40": len(t40)}
        chk[f"{res}_sigma_len"] = {"25": len(s25), "40": len(s40)}
    save_json(chk, os.path.join(ldir, "schedule_overlap_check.json"))
    print(json.dumps(chk, indent=2))


if __name__ == "__main__":
    main()
