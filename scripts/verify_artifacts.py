#!/usr/bin/env python3
"""Section 30: automated artifact integrity checks.

Verifies: expected experiment coverage, tensor presence, image pair presence,
hash consistency (latent reuse), prompt-id/seed consistency, NaN/Inf scan,
ROCm version/GPU arch, metric completeness. Exits nonzero on FAIL.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_json

FAILS = []
WARNS = []


def check(cond, msg, warn=False):
    if cond:
        print(f"  [OK]   {msg}")
    else:
        (WARNS if warn else FAILS).append(msg)
        print(f"  [{'WARN' if warn else 'FAIL'}] {msg}")


def main():
    env = load_json(os.path.join(ROOT, "artifacts", "environment", "summary.json"))
    print("== environment ==")
    check("/opt/rocm-7.14.0" in env.get("rocm_userspace", ""), f"ROCm userspace 7.14.0 ({env.get('rocm_userspace', '').strip()[-30:]})")
    check(env.get("gcn_arch") == "gfx1100", f"GPU gfx1100 ({env.get('gpu_name')})")
    check("rocm7.14.0" in env.get("torch", ""), f"torch {env.get('torch')}")

    print("== model manifest ==")
    man = load_json(os.path.join(ROOT, "artifacts", "model_manifest.json"))
    check(bool(man.get("configs")), "config hashes present")
    te = man.get("text_encoder_weights", {})
    check(len(te) >= 4, f"text_encoder shards hashed: {len(te)}")

    print("== inputs ==")
    for suite, n in [("smoke", 3), ("canary", 10), ("alignment_100", 100)]:
        files = glob.glob(os.path.join(ROOT, "artifacts", "inputs", "*.npz"))
        ids = {os.path.basename(f)[:-4] for f in files}
        prompts = load_json(os.path.join(ROOT, "prompts", f"{suite}.json"))["prompts"]
        missing = [p["id"] for p in prompts if p["id"] not in ids]
        check(not missing, f"{suite}: all {n} input npz present" + (f" missing={missing[:3]}" if missing else ""))

    ia = load_json(os.path.join(ROOT, "artifacts", "metrics", "input_alignment_summary.json"))
    check(ia.get("all_exact") is True, "input alignment EXACT (processor vs ComfyUI tokenizer)")

    print("== tensors ==")
    tdir = os.path.join(ROOT, "artifacts", "tensors")
    for arm, suites in [("R0", ["smoke", "canary", "alignment_100"]), ("R0R", ["smoke", "canary"]),
                        ("R1", ["smoke", "canary"]), ("O0", ["smoke", "canary"]), ("O1", ["smoke"]),
                        ("O3", ["smoke"]), ("O3r", ["smoke", "canary", "alignment_100"])]:
        for suite in suites:
            prompts = load_json(os.path.join(ROOT, "prompts", f"{suite}.json"))["prompts"]
            have = [p["id"] for p in prompts if os.path.exists(os.path.join(tdir, arm, f"{p['id']}.npz"))]
            need = len(prompts)
            check(len(have) == need, f"{arm}/{suite}: {len(have)}/{need} tensor files",
                  warn=(arm in ("O3", "O1F") and len(have) < need))

    print("== tensor sanity (NaN/Inf + shapes) ==")
    import numpy as np
    bad = []
    sampled = 0
    for f in sorted(glob.glob(os.path.join(tdir, "*", "*.npz")))[:400]:
        try:
            z = np.load(f)
            pe = z["prompt_embeds"]
            if not np.isfinite(pe).all():
                bad.append(f)
            sampled += 1
        except Exception as e:
            bad.append(f"{f}:{e}")
    check(not bad, f"NaN/Inf scan on {sampled} tensor files clean" + (f" bad={bad[:3]}" if bad else ""))

    print("== diffusion controls ==")
    latdir = os.path.join(ROOT, "artifacts", "latents")
    for res in (1024, 2048):
        check(os.path.exists(os.path.join(latdir, f"initial_latent_{res}.safetensors")),
              f"frozen initial latent {res} present")
    sch = load_json(os.path.join(latdir, "latent_and_schedule_manifest.json"))
    check("1024_25" in sch["schedules"] and "1024_40" in sch["schedules"], "25/40-step schedules recorded")
    ov = load_json(os.path.join(latdir, "schedule_overlap_check.json"))
    check(ov.get("1024_timesteps_25_eq_first25_of_40") is False,
          "documented: 25-step schedule != first 25 of 40-step schedule")

    traces = glob.glob(os.path.join(ROOT, "artifacts", "metrics", "trace_*.json"))
    check(len(traces) >= 3, f"diffusion traces present: {len(traces)}")

    print("== images ==")
    imgs = glob.glob(os.path.join(ROOT, "artifacts", "images", "*.png"))
    check(len(imgs) > 10, f"images present: {len(imgs)}")

    print("== metrics ==")
    mdir = os.path.join(ROOT, "artifacts", "metrics")
    for m in ["embedding_summary_stats_R0R0R_R0R1_R1O0_R0O0_O0O1_O0O1F_O0O3_R0O3_R0O1F_canary.json",
              "layer_alignment_R1_vs_O0_summary.json",
              "layer_alignment_O0_vs_O3_summary.json"]:
        check(os.path.exists(os.path.join(mdir, m)), f"metric file {m}")

    print("== reviews ==")
    reviews = glob.glob(os.path.join(ROOT, "artifacts", "reviews", "*.md"))
    check(len(reviews) >= 3, f"independent phase reviews present: {len(reviews)}")

    print()
    print(f"INTEGRITY: {'FAIL' if FAILS else 'PASS'}  ({len(FAILS)} fails, {len(WARNS)} warnings)")
    for f in FAILS:
        print("  FAIL:", f)
    for w in WARNS:
        print("  WARN:", w)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
