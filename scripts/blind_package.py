#!/usr/bin/env python3
"""Section 27: blind evaluation package generator.

Selects paired outputs (R0 vs O3r), randomizes left/right, hides arm identity.
Writes reports/blind_package/ with an index and a KEY file kept separately.
Does NOT fabricate any human results: evaluation marked pending until humans run it.
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", default="R0")
    ap.add_argument("--arm-b", default="O3r")
    ap.add_argument("--pattern", default="canary_*")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    imgdir = os.path.join(ROOT, "artifacts", "images")
    pairs = []
    for fa in sorted(glob.glob(os.path.join(imgdir, f"{args.pattern}_{args.arm_a}.png"))):
        fb = fa.replace(f"_{args.arm_a}.png", f"_{args.arm_b}.png")
        if os.path.exists(fb):
            pairs.append((fa, fb))
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.n]

    pkg = os.path.join(ROOT, "reports", "blind_package")
    os.makedirs(pkg, exist_ok=True)
    key = {}
    index = []
    for i, (fa, fb) in enumerate(pairs):
        tag = os.path.basename(fa).replace(f"_{args.arm_a}.png", "")
        left_is_a = rng.random() < 0.5
        la, lb = (fa, fb) if left_is_a else (fb, fa)
        fn_l, fn_r = f"pair_{i:03d}_left.png", f"pair_{i:03d}_right.png"
        shutil.copyfile(la, os.path.join(pkg, fn_l))
        shutil.copyfile(lb, os.path.join(pkg, fn_r))
        key[f"pair_{i:03d}"] = {"tag": tag, "left": args.arm_a if left_is_a else args.arm_b,
                                "right": args.arm_b if left_is_a else args.arm_a}
        index.append({"pair": f"pair_{i:03d}", "prompt_tag": tag})
    save_json(key, os.path.join(ROOT, "artifacts", "manifests", "blind_package_KEY.json"))
    save_json({"pairs": index, "instructions": (
        "For each pair choose for BOTH images: A better / B better / Tie on: "
        "prompt adherence, composition, text correctness, visual quality, overall. "
        "Record answers in answers.csv as pair,category,A/B/Tie."),
        "evaluation_status": "PENDING — no human results fabricated"},
        os.path.join(pkg, "index.json"))
    print(f"blind package: {len(pairs)} pairs in {pkg}; KEY stored separately (artifacts/manifests)")


if __name__ == "__main__":
    main()
