#!/usr/bin/env python3
"""v2 prompt-image semantic scoring with SigLIP (fixes the v1 gap).

For every generated image: sim(prompt, image) = cosine of normalized SigLIP
text/image embeddings. Raw pairwise similarity — never a softmax over a
single candidate. Produces per-arm distributions and paired deltas.

Model: google/siglip-so400m-patch14-384 (frozen revision recorded).

Inputs: images under /valdata/images/<group>/<pid>_<steps>s_<arm>.png
Outputs: artifacts/v2/metrics/semantic_scores_<group>.csv + summary json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

MODEL_ID = "google/siglip-so400m-patch14-384"


@torch.no_grad()
def embed_batch(model, processor, texts, images, device):
    ti = processor(text=texts, return_tensors="pt", padding="max_length",
                   max_length=64, truncation=True).to(device)
    ii = processor(images=images, return_tensors="pt").to(device)
    te = model.get_text_features(**ti)
    ie = model.get_image_features(**ii)
    te = torch.nn.functional.normalize(te, dim=-1).float().cpu()
    ie = torch.nn.functional.normalize(ie, dim=-1).float().cpu()
    return te, ie


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="population", help="image dir under /valdata/images")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--suite", default="alignment_100", help="prompt suite for texts")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    prompts = {p["id"]: p for p in C.load_prompts(args.suite)}
    files = sorted(glob.glob(os.path.join(C.VALDATA, "images", args.group, f"{args.pattern}.png")))
    rows = []
    from PIL import Image
    B = 16
    for i in range(0, len(files), B):
        batch_files = files[i:i + B]
        images = [Image.open(f).convert("RGB").resize((384, 384)) for f in batch_files]
        pids, armtags = [], []
        for f in batch_files:
            m = re.match(r"(.+)_(\d+)s_([A-Za-z0-9_]+)\.png", os.path.basename(f))
            pid, steps, arm = m.group(1), int(m.group(2)), m.group(3)
            pids.append(pid)
            armtags.append((pid, steps, arm))
        texts = [prompts[pid]["text"] for pid in pids]
        te, ie = embed_batch(model, processor, texts, images, device)
        for (pid, steps, arm), t, v in zip(armtags, te, ie):
            rows.append({"pid": pid, "category": prompts[pid].get("category", ""),
                         "steps": steps, "arm": arm,
                         "siglip_cosine": float((t * v).sum())})
        print(f"{i + len(batch_files)}/{len(files)}", flush=True)

    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    out_csv = os.path.join(mdir, f"semantic_scores_{args.group}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "category", "steps", "arm", "siglip_cosine"])
        w.writeheader()
        w.writerows(rows)

    # paired summary vs C_BF16
    by = {}
    for r in rows:
        by.setdefault((r["pid"], r["steps"], r["arm"]), r["siglip_cosine"])
    arms = sorted({r["arm"] for r in rows})
    summary = {a: C.stats_block([v for (p, s, ar), v in by.items() if ar == a]) for a in arms}
    paired = {}
    for a in arms:
        if a == "C_BF16":
            continue
        deltas = []
        for (p, s, ar), v in by.items():
            if ar != a:
                continue
            ref = by.get((p, s, "C_BF16"))
            if ref is not None:
                deltas.append(v - ref)
        if deltas:
            st = C.stats_block(deltas)
            import math
            n = st["n"]
            boots = []
            rng = np.random.default_rng(20261001)
            arr = np.asarray(deltas)
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                boots.append(float(arr[idx].mean()))
            st["boot_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            st["wilcoxon_p"] = None
            try:
                from scipy.stats import wilcoxon
                st["wilcoxon_p"] = float(wilcoxon(arr).pvalue)
            except Exception:
                pass
            paired[a] = st
    C.save_json({"model": MODEL_ID, "n_images": len(rows), "summary": summary,
                 "paired_delta_vs_C_BF16": paired},
                os.path.join(mdir, f"semantic_summary_{args.group}.json"))
    print("->", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
