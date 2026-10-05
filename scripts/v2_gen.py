#!/usr/bin/env python3
"""v2 shared diffusion engine: frozen-latent transplant generation on W7900D.

All v2 image experiments (population, crossover, perturbation, 2K) use this
module so the controls are literally identical code paths:
  - one QwenImage21Pipeline instance (bf16 transformer + VAE on cuda)
  - frozen initial latent (SHA256-checked against the manifest)
  - frozen 25/40-step schedules (asserted against the manifest)
  - true_cfg_scale=1.0, no negative prompt, no CFG
  - only the positive prompt embedding varies between arms
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

from safetensors.torch import load_file

_PIPE = None
_LAT = {}
_SCHED = None


def get_pipe():
    global _PIPE
    if _PIPE is None:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
        pipe = QwenImage21Pipeline.from_pretrained(C.model_snapshot_path(), torch_dtype=torch.bfloat16)
        pipe.text_encoder = None
        pipe.processor = None
        pipe.to("cuda")
        _PIPE = pipe
    return _PIPE


def get_latent(resolution: int, suffix: str | None = None) -> torch.Tensor:
    key = (resolution, suffix or "")
    if key not in _LAT:
        p = os.path.join(C.VALDATA, "latents", f"v2_initial_latent_{resolution}{key[1]}.safetensors")
        _LAT[key] = load_file(p)["latent"]
    return _LAT[key]


def get_sched_manifest() -> dict:
    global _SCHED
    if _SCHED is None:
        with open(os.path.join(C.VALDATA, "latents", "v2_latent_schedule_manifest.json")) as f:
            _SCHED = json.load(f)
    return _SCHED


def get_embeds(arm: str, pid: str) -> torch.Tensor:
    z = np.load(os.path.join(C.VALDATA, "tensors", arm, f"{pid}.npz"), allow_pickle=True)
    e = torch.from_numpy(z["cond"].astype(np.float32))
    if e.ndim == 2:
        e = e.unsqueeze(0)
    return e


def generate(pipe, pos_embeds: torch.Tensor, steps: int, resolution: int,
             latent: torch.Tensor | None = None, record_trace: bool = False):
    """One controlled generation. Returns (pil_image, final_latent_np, trace|None)."""
    lat_cpu = latent if latent is not None else get_latent(resolution, None)
    lat = lat_cpu.to("cuda", dtype=torch.bfloat16).clone()
    trace = {"latents": [], "model_out": []} if record_trace else None

    hook = None
    if record_trace:
        def _hook(module, args, kwargs, output):
            o = output[0] if isinstance(output, tuple) else output
            trace["model_out"].append(o.detach()[:, -lat.size(1):].float().cpu().numpy())
            return output
        hook = pipe.transformer.register_forward_hook(_hook, with_kwargs=True)

    lat_holder = {}

    def cb(p, i, t, cbk):
        if record_trace:
            trace["latents"].append(cbk["latents"].detach().float().cpu().numpy())
        else:
            lat_holder["final"] = cbk["latents"].detach().float().cpu().numpy()
        return {}

    try:
        out = pipe(
            prompt=None,
            prompt_embeds=pos_embeds.to("cuda", dtype=torch.bfloat16),
            prompt_embeds_mask=None,
            num_inference_steps=steps,
            height=resolution, width=resolution,
            latents=lat,
            true_cfg_scale=1.0,
            negative_prompt=None,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pil",
        )
    finally:
        if hook is not None:
            hook.remove()
    final_lat = (trace["latents"][-1] if record_trace else lat_holder.get("final"))
    return out.images[0], final_lat, trace


def assert_schedule(pipe, resolution: int, steps: int):
    ref = get_sched_manifest()["schedules"][f"{resolution}_{steps}"]
    ts = pipe.scheduler.timesteps.detach().cpu().tolist()
    assert np.allclose(ts, ref["timesteps"]), "timesteps drifted from frozen manifest"


def controlled_run(arm_embeds: torch.Tensor, steps: int, resolution: int,
                   latent: torch.Tensor | None = None, record_trace: bool = False):
    """Schedule-asserted controlled run; returns (img, final_latent, trace)."""
    pipe = get_pipe()
    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    torch.cuda.synchronize()
    t0 = __import__("time").time()
    img, final_lat, trace = generate(pipe, arm_embeds, steps, resolution, latent, record_trace)
    torch.cuda.synchronize()
    dt = __import__("time").time() - t0
    assert_schedule(pipe, resolution, steps)
    return img, final_lat, trace, dt
