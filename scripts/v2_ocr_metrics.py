#!/usr/bin/env python3
"""v2 OCR / text-rendering metrics (English, Chinese, mixed).

For every generated image whose prompt carries an expected `text_target`
(pipelines suites mark them via category text_en/text_zh/mixed) or embeds a
quoted string, run RapidOCR and score:
  - exact match of primary target string
  - CER / WER vs primary target
Primary target = first quoted string in the prompt (decorative extras ignored).

Outputs: artifacts/v2/metrics/ocr_scores_<group>.csv + summary json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C


def primary_target(text: str) -> str | None:
    m = re.search(r'[""«"\'\u201c](.+?)[""»"\'\u201d]', text)
    return m.group(1) if m else None


def cer(ref: str, hyp: str) -> float:
    import editdistance
    ref_c, hyp_c = list(ref), list(hyp)
    if not ref_c:
        return 0.0
    return editdistance.eval(ref_c, hyp_c) / len(ref_c)


def wer(ref: str, hyp: str) -> float:
    import editdistance
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    return editdistance.eval(r, h) / len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="population")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--suite", default="alignment_100")
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    prompts = {p["id"]: p for p in C.load_prompts(args.suite)}
    files = sorted(glob.glob(os.path.join(C.VALDATA, "images", args.group, f"{args.pattern}.png")))
    rows = []
    for f in files:
        base = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(\d+)s_([A-Za-z0-9_]+)$", base)
        if not m:
            continue
        pid, steps, arm = m.group(1), int(m.group(2)), m.group(3)
        p = prompts.get(pid)
        if p is None:
            continue
        target = p.get("text_target") or primary_target(p["text"])
        if not target:
            continue
        res, _ = ocr(f)
        ocr_text = " ".join(x[1] for x in res) if res else ""
        rows.append({
            "pid": pid, "category": p.get("category", ""), "steps": steps, "arm": arm,
            "target": target, "ocr_text": ocr_text,
            "exact": int(target.strip().upper() == ocr_text.strip().upper()),
            "contains": int(target.strip().upper() in ocr_text.strip().upper()),
            "cer": round(cer(target, ocr_text), 4), "wer": round(wer(target, ocr_text), 4)})
        print(base, "exact=", rows[-1]["exact"], "cer=", rows[-1]["cer"], flush=True)

    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    out_csv = os.path.join(mdir, f"ocr_scores_{args.group}.csv")
    with open(out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()) if rows else
                           ["pid", "category", "steps", "arm", "target", "ocr_text",
                            "exact", "contains", "cer", "wer"])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"n": len(sub), "exact_rate": sum(r["exact"] for r in sub) / len(sub),
                        "contains_rate": sum(r["contains"] for r in sub) / len(sub),
                        "mean_cer": sum(r["cer"] for r in sub) / len(sub)}
    C.save_json({"ocr_engine": "rapidocr_onnxruntime", "summary_vs_target": summary},
                os.path.join(mdir, f"ocr_summary_{args.group}.json"))
    print("->", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
