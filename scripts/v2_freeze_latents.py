#!/usr/bin/env python3
"""v2: freeze initial latents + exact 25/40-step schedules for 1024/2048.

Latents generated once on CPU (device-independent), saved as safetensors with
SHA256; schedules replicate QwenImage21Pipeline.__call__ construction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

from safetensors.torch import save_file
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

SEED = 20261001
EXTRA_SEEDS = [20261002, 20261003]  # 3-seed subset support (§46)


def main() -> int:
    snap = C.model_snapshot_path()
    ldir = os.path.join(C.VALDATA, "latents")
    os.makedirs(ldir, exist_ok=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
    manifest = {"seed": SEED, "latents": {}, "schedules": {}}
    for res in (1024, 2048):
        for seed in [SEED] + EXTRA_SEEDS:
            suffix = "" if seed == SEED else f"_s{seed}"
            g = torch.Generator(device="cpu").manual_seed(seed)
            latent = torch.randn((1, 64, res // 16, res // 16), generator=g, dtype=torch.float32)
            seq = (res // 16) * (res // 16)
            packed = latent.reshape(1, 64, seq).permute(0, 2, 1).contiguous()  # [1, seq, 64] packed
            p = os.path.join(ldir, f"v2_initial_latent_{res}{suffix}.safetensors")
            save_file({"latent": packed}, p)
            manifest["latents"][f"{res}{suffix}"] = {"path": p, "sha256": C.sha256_file(p),
                                                     "tensor_sha256": C.tensor_sha256(packed),
                                                     "shape": list(packed.shape), "seed": seed}
        seq = (res // 16) * (res // 16)
        for steps in (25, 40):
            sigmas = np.linspace(1.0, 1 / steps, steps)
            mu = calculate_shift(seq, sched.config.get("base_image_seq_len", 256),
                                 sched.config.get("max_image_seq_len", 4096),
                                 sched.config.get("base_shift", 0.5),
                                 sched.config.get("max_shift", 1.15))
            s2 = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
            timesteps, _ = retrieve_timesteps(s2, steps, "cpu", sigmas=sigmas, mu=mu)
            manifest["schedules"][f"{res}_{steps}"] = {
                "seq_len": seq, "mu": float(mu), "timesteps": timesteps.tolist(),
                "sigmas": sigmas.tolist()}
    out = os.path.join(ldir, "v2_latent_schedule_manifest.json")
    C.save_json(manifest, out)
    print("->", out)
    print("overlap 1024_25 vs first25 of 1024_40:",
          manifest["schedules"]["1024_25"]["timesteps"] ==
          manifest["schedules"]["1024_40"]["timesteps"][:25])
    return 0


if __name__ == "__main__":
    sys.exit(main())
