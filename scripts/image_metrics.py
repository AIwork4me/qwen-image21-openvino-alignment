#!/usr/bin/env python3
"""Phase I: image-level metrics between paired generations.

Levels: pixel (PSNR/SSIM), perceptual (LPIPS), semantic (CLIPScore), text rendering (OCR).
Inputs: paired images artifacts/images/<tag>_<armA>.png vs <tag>_<armB>.png.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json, stats_block

from PIL import Image


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def lpips_score(model, a, b):
    import torch
    ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    return float(model(ta, tb).item())


def clip_score(texts_a, imgs, model, processor, device="cpu"):
    import torch
    scores = []
    for txt, img in zip(texts_a, imgs):
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(device)
        out = model(**inputs)
        logits = out.logits_per_image[0].softmax(dim=0)
        scores.append(float(logits.max().item()))
    return scores


def normalize_text(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def cer(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    import editdistance
    return editdistance.distance(r, h) / len(r)


def exact_str(ref, hyp):
    return float(normalize_text(ref) == normalize_text(hyp))


def ocr_image(path, engine=None):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    res, _ = _OCR(path)
    if not res:
        return ""
    return " ".join(line[1] for line in res)


_OCR = None


def add_ocr_metrics(rows, prompts_by_tag):
    """Add OCR text-extraction + CER/exact-match vs expected text (for text-rendering prompts)."""
    for r in rows:
        p = prompts_by_tag.get(r.get("pid", r["tag"]))
        if p is None or p.get("category") not in ("text_en", "text_zh", "mixed_text", "signage", "poster", "label"):
            continue
        fa = os.path.join(ROOT, "artifacts", "images", f"{r['tag']}_{r['arm_a']}.png")
        fb = os.path.join(ROOT, "artifacts", "images", f"{r['tag']}_{r['arm_b']}.png")
        ta, tb = ocr_image(fa), ocr_image(fb)
        exp = " ".join(x for tup in re.findall(r'"([^"]+)"|\u201c([^\u201d]+)\u201d', p["text"]) for x in tup if x)
        r["ocr_a"] = ta
        r["ocr_b"] = tb
        r["expected_text"] = exp
        r["cer_a"] = cer(exp, ta) if exp else None
        r["cer_b"] = cer(exp, tb) if exp else None
        r["exact_a"] = exact_str(exp, ta) if exp else None
        r["exact_b"] = exact_str(exp, tb) if exp else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*", help="tag glob for image pairs")
    ap.add_argument("--arm-a", default="R0")
    ap.add_argument("--arm-b", default="O3")
    ap.add_argument("--suite", default=None, help="prompt suite for text/CLIP metadata")
    ap.add_argument("--ocr", action="store_true")
    args = ap.parse_args()

    imgdir = os.path.join(ROOT, "artifacts", "images")
    pairs = []
    for fa in sorted(glob.glob(os.path.join(imgdir, f"{args.pattern}_{args.arm_a}.png"))):
        fb = fa.replace(f"_{args.arm_a}.png", f"_{args.arm_b}.png")
        if os.path.exists(fb):
            pairs.append((fa, fb))
    if not pairs:
        print("no image pairs found for", args.pattern, args.arm_a, args.arm_b)
        return

    lpips_model = None
    try:
        import lpips as lpips_lib
        import torch
        lpips_model = lpips_lib.LPIPS(net="alex", verbose=False)
    except Exception as e:
        print("LPIPS unavailable:", e)

    rows = []
    for fa, fb in pairs:
        tag = os.path.basename(fa).replace(f"_{args.arm_a}.png", "")
        a = np.asarray(Image.open(fa).convert("RGB"))
        b = np.asarray(Image.open(fb).convert("RGB"))
        row = {"tag": tag, "arm_a": args.arm_a, "arm_b": args.arm_b,
               "psnr": psnr(a, b), "ssim": ssim(a, b) if a.shape == b.shape else None}
        if lpips_model is not None and a.shape == b.shape:
            row["lpips"] = lpips_score(lpips_model, a, b)
        rows.append(row)
        print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    mdir = os.path.join(ROOT, "artifacts", "metrics")
    os.makedirs(mdir, exist_ok=True)
    if args.ocr:
        prompts_by_tag = {}
        for suite_file in glob.glob(os.path.join(ROOT, "prompts", "*.json")):
            for p in json.load(open(suite_file))["prompts"]:
                prompts_by_tag[p["id"]] = p
        # tag -> pid mapping: tags like "canary_C09_s20261001_40s"
        for r in rows:
            m = re.match(r"[a-z0-9_]+?_([A-Z]\d+)_s(\d+)_(\d+)s", r["tag"])
            if m:
                r["pid"] = m.group(1)
                r["seed"] = int(m.group(2))
                r["steps"] = int(m.group(3))
                r["prompt"] = prompts_by_tag.get(m.group(1), {}).get("text")
        rows = add_ocr_metrics(rows, {r.get("pid"): {"category": (prompts_by_tag.get(r.get("pid")) or {}).get("category"),
                                                     "text": r.get("prompt")} for r in rows})
    allkeys = sorted({k for r in rows for k in r})
    out = os.path.join(mdir, f"image_metrics_{re.sub(r'[^A-Za-z0-9_.-]', '_', args.pattern)}_{args.arm_a}_vs_{args.arm_b}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allkeys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {k: stats_block([r[k] for r in rows if r.get(k) is not None])
               for k in ("psnr", "ssim", "lpips") if any(r.get(k) is not None for r in rows)}
    save_json({"pattern": args.pattern, "arm_a": args.arm_a, "arm_b": args.arm_b,
               "n_pairs": len(rows), "summary": summary}, out.replace(".csv", "_summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
