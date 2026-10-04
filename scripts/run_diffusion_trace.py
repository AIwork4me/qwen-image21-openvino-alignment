#!/usr/bin/env python3
"""Phase E/F workhorse: run the W7900 diffusion pipeline with INJECTED prompt embeddings.

Both arms (ROCm-derived and OpenVINO-derived embeddings) run through the SAME loaded
pipeline in one process. The only intentional variable is the positive prompt embedding.
All other inputs are asserted identical: initial latent (bitwise, from safetensors),
timesteps/sigmas (compared to the frozen manifest), scheduler config, resolution.
Qwen-Image 2.1 official sampling: true_cfg_scale=1 (no CFG, no negative prompt).

Captures per-step: latent (after scheduler step), transformer noise_pred.
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


def load_pipe(cfg):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    snap = cfg["model"]["snapshot_path"]
    pipe = QwenImage21Pipeline.from_pretrained(snap, torch_dtype=torch.bfloat16)
    pipe.text_encoder = None
    pipe.processor = None
    pipe.to("cuda")
    return pipe


def get_embeds(arm, pid):
    p = os.path.join(ROOT, "artifacts", "tensors", arm, f"{pid}.npz")
    z = np.load(p)
    e = torch.from_numpy(z["prompt_embeds_bf16_view"].astype(np.float32))
    if e.ndim == 2:
        e = e.unsqueeze(0)
    return e


def capture_run(pipe, lat_cpu, pos_embeds, steps, resolution, save_tensors):
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
            height=resolution,
            width=resolution,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--pid", default="S01")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--arm-a", default="R0")
    ap.add_argument("--arm-b", default="O3")
    ap.add_argument("--arms", default=None, help="comma list overriding pair mode (multi-arm)")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--save-tensors", action="store_true")
    ap.add_argument("--seed-tag", default="")
    args = ap.parse_args()

    cfg = load_cfg()
    pipe = load_pipe(cfg)
    lat = load_file(os.path.join(ROOT, "artifacts", "latents", f"initial_latent_{args.resolution}.safetensors"))["latent"]
    lat_hash = tensor_hash(lat.numpy())
    sched_manifest = json.load(open(os.path.join(ROOT, "artifacts", "latents", "latent_and_schedule_manifest.json")))
    sched_ref = sched_manifest["schedules"][f"{args.resolution}_{args.steps}"]

    arms = args.arms.split(",") if args.arms else [args.arm_a, args.arm_b]
    if len(arms) == 2 and arms[0] == arms[1]:
        arms = [arms[0], arms[0] + "#2"]  # avoid vacuous self-comparison from dict keying
    results = {}
    for arm in arms:
        pe = get_embeds(arm, args.pid)
        pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
        torch.cuda.synchronize(); t0 = time.time()
        img, trace = capture_run(pipe, lat, pe, args.steps, args.resolution, args.save_tensors)
        torch.cuda.synchronize()
        gen_s = time.time() - t0
        n_steps = len(trace["latents"])
        ts = pipe.scheduler.timesteps.detach().cpu().tolist() if len(pipe.scheduler.timesteps) else []
        assert len(trace["model_out"]) == n_steps, (len(trace["model_out"]), n_steps)
        key = arm.replace("#2", "")
        results[arm] = {"img": img, "trace": trace, "gen_s": gen_s,
                        "pe_hash": tensor_hash(pe.numpy()), "n_steps": n_steps}
        print(f"[{arm}] generated {gen_s:.1f}s steps={n_steps} "
              f"timesteps_match_manifest={np.allclose(ts, sched_ref['timesteps']) if ts else 'n/a'}")

    step_stats = []
    ref = arms[0]
    for arm in arms[1:]:
        ta, tb = results[ref]["trace"], results[arm]["trace"]
        n = min(len(ta["latents"]), len(tb["latents"]))
        for i in range(n):
            lm = paired_metrics(ta["latents"][i], tb["latents"][i])
            mo = paired_metrics(ta["model_out"][i], tb["model_out"][i])
            step_stats.append({"vs": arm, "step": i, "latent.cosine": lm["cosine"],
                               "latent.rel_l2": lm["rel_l2"], "latent.rmse": lm["rmse"],
                               "latent.max_abs": lm["max_abs"],
                               "model_out.cosine": mo["cosine"], "model_out.rel_l2": mo["rel_l2"]})

    imgdir = os.path.join(ROOT, "artifacts", "images")
    os.makedirs(imgdir, exist_ok=True)
    tag = args.seed_tag or f"{args.pid}_{args.steps}s"
    for arm in arms:
        results[arm]["img"].save(os.path.join(imgdir, f"{tag}_{arm}.png"))

    save_json({
        "suite": args.suite, "pid": args.pid, "steps": args.steps, "resolution": args.resolution,
        "arms": arms,
        "controls": {"latent_hash": lat_hash,
                     "pe_hashes": {arm: results[arm]["pe_hash"] for arm in arms},
                     "timesteps": sched_ref["timesteps"], "mu": sched_ref["mu"],
                     "true_cfg": 1.0, "negative": None},
        "gen_s": {arm: results[arm]["gen_s"] for arm in arms},
        "n_steps": {arm: results[arm]["n_steps"] for arm in arms},
        "step_stats": step_stats,
        "final": {r["vs"]: {"latent.cosine": r["latent.cosine"], "latent.rel_l2": r["latent.rel_l2"]}
                  for r in step_stats if r["step"] == max((x["step"] for x in step_stats), default=0)},
    }, os.path.join(ROOT, "artifacts", "metrics", f"trace_{tag}.json"))

    if args.save_tensors:
        tdir = os.path.join(ROOT, "artifacts", "tensors", "traces")
        os.makedirs(tdir, exist_ok=True)
        np.savez_compressed(os.path.join(tdir, f"{tag}_latents.npz"),
                            **{arm: np.stack(results[arm]["trace"]["latents"]) for arm in arms})
    for s in step_stats[-len(arms):]:
        print(f"final vs {s['vs']}: latent cos {s['latent.cosine']:.6f} rel_l2 {s['latent.rel_l2']:.6f}")


if __name__ == "__main__":
    main()
