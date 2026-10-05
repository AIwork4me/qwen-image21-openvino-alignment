#!/usr/bin/env python3
"""v2 crossover experiment: when does quantized-conditioning trajectory error
become locked in?

For quant arm Q in {C_INT8, C_W4A8} and crossover step k in {5,10,15,20,25,30}
at 40 steps / 1024px with the frozen latent:

  cells (each a full denoising trajectory with cond possibly swapped at k):
    PURE_BF16      BF16 cond for all 40 steps                 (kv_cache off)
    PURE_Q         Q cond for all 40 steps                    (kv_cache off)
    B_k            BF16 cond steps [0,k), Q cond steps [k,40)
    C_k            Q cond steps [0,k), BF16 cond steps [k,40)

kv_cache is disabled for every cell (cond swap must invalidate cached K/V),
so all four cells share identical math settings.

Final-latent rel_l2 vs PURE_BF16 per cell -> lock-in analysis.

Outputs: artifacts/v2/metrics/v2_crossover.json (+ per-run images/latents).
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
import v2_gen as G

from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps


def manual_run(pipe, pe_first, pe_second, k, steps, resolution, latent):
    """Manual faithful denoising loop (kv_cache=None).

    steps [0,k) use pe_first, steps [k,steps) use pe_second.
    k=None -> pe_first everywhere; k=0 -> pe_second everywhere.
    """
    lat = latent.to("cuda", dtype=torch.bfloat16).clone()
    h = w = resolution // 16
    img_shapes = [[(1, h, w)]]
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(lat.shape[1], pipe.scheduler.config.get("base_image_seq_len", 256),
                         pipe.scheduler.config.get("max_image_seq_len", 4096),
                         pipe.scheduler.config.get("base_shift", 0.5),
                         pipe.scheduler.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, "cuda", sigmas=sigmas, mu=mu)
    ref = G.get_sched_manifest()["schedules"][f"{resolution}_{steps}"]
    assert np.allclose(timesteps.detach().cpu().tolist(), ref["timesteps"]), "schedule drift"
    pipe.scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        use_second = (k is not None and i >= k)
        pe = pe_second if use_second else pe_first
        timestep = t.expand(lat.shape[0]).to(lat.dtype)
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=lat, timestep=timestep / 1000,
                encoder_hidden_states=pe.to("cuda", dtype=torch.bfloat16),
                encoder_hidden_states_mask=None,
                img_shapes=img_shapes, img_mask=None,
                attention_kwargs=None, kv_cache=None, return_dict=False)[0]
        noise_pred = noise_pred[:, -lat.size(1):]
        lat = pipe.scheduler.step(noise_pred, t, lat, return_dict=False)[0]
    return lat.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="C09")
    ap.add_argument("--quant-arms", default="C_INT8,C_W4A8")
    ap.add_argument("--k-points", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    C.gate_environment()
    pipe = G.get_pipe()
    latent = G.get_latent(args.resolution)
    pe_bf16 = G.get_embeds("C_BF16", args.pid)
    imgdir = os.path.join(C.VALDATA, "images", "crossover")
    os.makedirs(imgdir, exist_ok=True)

    results = {}
    for qa in args.quant_arms.split(","):
        pe_q = G.get_embeds(qa, args.pid)
        cells = {}
        # pure cells: k=None keeps pe_first everywhere
        pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
        t0 = time.time()
        cells["PURE_BF16"] = manual_run(pipe, pe_bf16, pe_bf16, None, args.steps, args.resolution, latent)
        pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
        cells[f"PURE_{qa}"] = manual_run(pipe, pe_q, pe_q, None, args.steps, args.resolution, latent)
        # B_k: BF16 first k steps then quant; C_k: quant first k steps then BF16
        for k in args.k_points:
            pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
            cells[f"B_k{k}"] = manual_run(pipe, pe_bf16, pe_q, k, args.steps, args.resolution, latent)
            pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
            cells[f"C_k{k}"] = manual_run(pipe, pe_q, pe_bf16, k, args.steps, args.resolution, latent)
        ref = torch.from_numpy(cells["PURE_BF16"])
        rows = {}
        for name, lat in cells.items():
            rows[name] = {"latent_rel_l2_vs_pure_bf16": C.rel_l2(ref, torch.from_numpy(lat)),
                          "latent_cos_vs_pure_bf16": C.cosine(ref, torch.from_numpy(lat))}
        results[qa] = {"cells_s": round(time.time() - t0, 1), "rows": rows,
                       "pe_sha256": {a: C.tensor_sha256(G.get_embeds(a, args.pid))
                                     for a in ("C_BF16", qa)}}
        print(qa, json.dumps({k: round(v["latent_rel_l2_vs_pure_bf16"], 5)
                              for k, v in rows.items()}, indent=1), flush=True)
        np.save(os.path.join(imgdir, f"{args.pid}_{args.steps}s_pure_bf16_nocache.npy"), cells["PURE_BF16"])

    C.save_json({
        "experiment_id": f"V2-CROSSOVER-{args.pid}-{args.steps}",
        "pid": args.pid, "steps": args.steps, "resolution": args.resolution,
        "k_points": args.k_points, "kv_cache": "disabled (cond swap correctness)",
        "controls": {"latent_sha256": C.tensor_sha256(latent), "true_cfg": 1.0},
        "results": results,
    }, os.path.join(C.REPO, "artifacts", "v2", "metrics", f"v2_crossover_{args.pid}_{args.steps}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
