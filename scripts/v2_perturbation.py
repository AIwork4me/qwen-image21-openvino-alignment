#!/usr/bin/env python3
"""v2 perturbation calibration: is the quantization-error *direction* unusually
harmful, or is the diffusion backend generically sensitive to conditioning
noise of the same magnitude?

delta_Q = E_Q - E_BF16  (per prompt, frozen tokenization)

Conditions (all 25 and 40 steps, frozen latent):
  E0        baseline: E_BF16 unchanged
  0.25x     E_BF16 + 0.25 * delta
  0.5x      E_BF16 + 0.5  * delta
  1x        E_BF16 + 1.0  * delta   (== E_Q, verified by hash)
  2x        E_BF16 + 2.0  * delta
  R_gN      global-norm-matched random perturbation (seeded; N in {1..5})
  R_tN      per-token-norm-matched random perturbation (seeded; N in {1..3})

Final-latent rel_l2 vs E0 baseline per condition.

Outputs: artifacts/v2/metrics/v2_perturbation_{pid}.json + images.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="C09")
    ap.add_argument("--quant-arms", default="C_INT8,C_W4A8")
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 40])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--n-global", type=int, default=5)
    ap.add_argument("--n-token", type=int, default=3)
    args = ap.parse_args()

    C.gate_environment()
    imgdir = os.path.join(C.VALDATA, "images", "perturbation")
    os.makedirs(imgdir, exist_ok=True)
    latent = G.get_latent(args.resolution)
    pe0 = G.get_embeds("C_BF16", args.pid)

    out = {"experiment_id": f"V2-PERTURBATION-{args.pid}", "pid": args.pid,
           "resolution": args.resolution, "steps": args.steps,
           "controls": {"latent_sha256": C.tensor_sha256(latent), "true_cfg": 1.0},
           "conditions": {}}

    for steps in args.steps:
        base_img, base_lat, _, base_dt = G.controlled_run(pe0, steps, args.resolution)
        base_img.save(os.path.join(imgdir, f"{args.pid}_{steps}s_E0.png"))
        out["conditions"][f"steps{steps}"] = {"E0": {"gen_s": round(base_dt, 2)}}
        base_t = torch.from_numpy(base_lat)

        for qa in args.quant_arms.split(","):
            pe_q = G.get_embeds(qa, args.pid)
            delta = pe_q - pe0
            gnorm = delta.norm()
            rows = {}
            for scale in (0.25, 0.5, 1.0, 2.0):
                pe = pe0 + scale * delta
                if scale == 1.0:
                    assert C.tensor_sha256(pe) == C.tensor_sha256(pe_q), "1x delta must equal E_Q"
                _, flat, _, dt = G.controlled_run(pe, steps, args.resolution)
                rows[f"{scale}x"] = {"rel_l2": C.rel_l2(base_t, torch.from_numpy(flat)),
                                     "gen_s": round(dt, 2)}
            # global-norm-matched random
            for n in range(1, args.n_global + 1):
                g = torch.Generator().manual_seed(910_000 + n)
                r = torch.randn(delta.shape, generator=g)
                r = r * (gnorm / r.norm())
                pe = pe0 + r
                _, flat, _, dt = G.controlled_run(pe, steps, args.resolution)
                rows[f"R_g{n}"] = {"rel_l2": C.rel_l2(base_t, torch.from_numpy(flat)),
                                   "gen_s": round(dt, 2)}
            # per-token-norm-matched random
            tn = delta.norm(dim=-1, keepdim=True)
            for n in range(1, args.n_token + 1):
                g = torch.Generator().manual_seed(920_000 + n)
                r = torch.randn(delta.shape, generator=g)
                r = r * (tn / r.norm(dim=-1, keepdim=True))
                pe = pe0 + r
                _, flat, _, dt = G.controlled_run(pe, steps, args.resolution)
                rows[f"R_t{n}"] = {"rel_l2": C.rel_l2(base_t, torch.from_numpy(flat)),
                                   "gen_s": round(dt, 2)}
            out["conditions"][f"steps{steps}"][qa] = {
                "delta_norm": float(gnorm),
                "delta_rel_norm_vs_e0": C.rel_l2(pe0, pe_q),
                "rows": rows}
            print(f"[{args.pid} {steps}s {qa}]", json.dumps(
                {k: round(v["rel_l2"], 4) for k, v in rows.items()}), flush=True)

    C.save_json(out, os.path.join(C.REPO, "artifacts", "v2", "metrics",
                                  f"v2_perturbation_{args.pid}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
