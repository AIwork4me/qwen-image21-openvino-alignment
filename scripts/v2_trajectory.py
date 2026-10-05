#!/usr/bin/env python3
"""v2 embedding transplant + full trajectory tracing on the W7900 diffusion backend.

Same pipeline instance for all arms; only the positive prompt embedding varies.
Controls asserted: identical frozen latent (bitwise), identical scheduler/
timesteps (vs manifest), resolution, no CFG (true_cfg_scale=1), no negative
prompt.  Captures per-step latent + transformer prediction and final image.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

from safetensors.torch import load_file


def load_pipe():
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(C.model_snapshot_path(), torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    pipe.processor = None
    pipe.to("cuda")
    return pipe


def get_embeds(arm: str, pid: str) -> torch.Tensor:
    z = np.load(os.path.join(C.VALDATA, "tensors", arm, f"{pid}.npz"), allow_pickle=True)
    e = torch.from_numpy(z["cond"].astype(np.float32))
    if e.ndim == 2:
        e = e.unsqueeze(0)
    return e


def capture_run(pipe, lat_cpu, pos_embeds, steps, resolution):
    lat = lat_cpu.to("cuda", dtype=torch.bfloat16).clone()
    trace = {"latents": [], "model_out": []}

    def hook(module, args, kwargs, output):
        o = output[0] if isinstance(output, tuple) else output
        trace["model_out"].append(o.detach()[:, -lat.size(1):].float().cpu().numpy())
        return output

    handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)

    def cb(p, i, t, cbk):
        trace["latents"].append(cbk["latents"].detach().float().cpu().numpy())
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
        handle.remove()
    return out.images[0], trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--pids", default="S01,S02")
    ap.add_argument("--arms", default="C_BF16,C_INT8,C_W4A8")
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 40])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--outdir-tag", default="")
    args = ap.parse_args()

    C.gate_environment()
    pipe = load_pipe()
    lat = load_file(os.path.join(C.VALDATA, "latents", f"v2_initial_latent_{args.resolution}.safetensors"))["latent"]
    sched_manifest = json.load(open(os.path.join(C.VALDATA, "latents", "v2_latent_schedule_manifest.json")))

    imgdir = os.path.join(C.VALDATA, "images", "trajectory")
    os.makedirs(imgdir, exist_ok=True)
    metrics_dir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    arms = args.arms.split(",")
    pids = args.pids.split(",")

    for pid in pids:
        for steps in args.steps:
            sched_ref = sched_manifest["schedules"][f"{args.resolution}_{steps}"]
            results = {}
            for arm in arms:
                pe = get_embeds(arm, pid)
                pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
                torch.cuda.synchronize(); t0 = time.time()
                img, trace = capture_run(pipe, lat, pe, steps, args.resolution)
                torch.cuda.synchronize()
                ts = pipe.scheduler.timesteps.detach().cpu().tolist()
                assert len(trace["model_out"]) == len(trace["latents"])
                assert np.allclose(ts, sched_ref["timesteps"]), "timesteps drifted from frozen manifest"
                results[arm] = {"img": img, "trace": trace, "gen_s": time.time() - t0,
                                "pe_hash": C.tensor_sha256(pe)}
                img.save(os.path.join(imgdir, f"{pid}_{steps}s_{arm}.png"))
                print(f"[{pid} {steps}s {arm}] {results[arm]['gen_s']:.1f}s steps={len(trace['latents'])}", flush=True)

            ref = arms[0]
            step_stats = []
            for arm in arms[1:]:
                ta, tb = results[ref]["trace"], results[arm]["trace"]
                n = min(len(ta["latents"]), len(tb["latents"]))
                for i in range(n):
                    la = torch.from_numpy(ta["latents"][i]); lb = torch.from_numpy(tb["latents"][i])
                    ma = torch.from_numpy(ta["model_out"][i]); mb = torch.from_numpy(tb["model_out"][i])
                    step_stats.append({
                        "vs": arm, "step": i,
                        "latent_rel_l2": C.rel_l2(la, lb), "latent_cos": C.cosine(la, lb),
                        "model_rel_l2": C.rel_l2(ma, mb), "model_cos": C.cosine(ma, mb)})
            tag = f"{pid}_{steps}s"
            C.save_json({
                "experiment_id": f"V2-TRAJ-{pid}-{steps}", "suite": args.suite, "pid": pid,
                "steps": steps, "resolution": args.resolution, "arms": arms,
                "controls": {"latent_sha256": C.tensor_sha256(lat),
                             "pe_hashes": {a: results[a]["pe_hash"] for a in arms},
                             "timesteps": sched_ref["timesteps"], "mu": sched_ref["mu"],
                             "true_cfg": 1.0, "negative": None},
                "gen_s": {a: results[a]["gen_s"] for a in arms},
                "step_stats": step_stats,
                "final": {r["vs"]: {"latent_rel_l2": r["latent_rel_l2"], "latent_cos": r["latent_cos"]}
                          for r in step_stats if r["step"] == max(x["step"] for x in step_stats)},
            }, os.path.join(metrics_dir, f"v2_trajectory_{tag}.json"))
            print(f"[{pid} {steps}s] final:", {r["vs"]: round(r["latent_rel_l2"], 4) for r in step_stats if r["step"] == max(x["step"] for x in step_stats)}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
