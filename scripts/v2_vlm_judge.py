#!/usr/bin/env python3
"""v2 VLM pairwise blind judge: Qwen3-VL-8B (official snapshot weights) A/B.

For image pairs (BF16, quant) of the same pid/steps/seed:
  - randomize A/B order (seeded, recorded)
  - ask: which image better matches the prompt (prompt adherence), or tie
  - order-swap consistency check on a subset (same pair, swapped question)

Judge = Qwen/Qwen-Image-2.1 text_encoder (Qwen3-VL-8B-Instruct, fp16 GPU),
the official full VL model — reproducible; revision recorded. Note honest
caveat: this judge shares weights with the encoder under validation (only
the quantized arms differ); recorded in reports.

Outputs: artifacts/v2/metrics/vlm_judge_<group>.csv + summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

JUDGE_PROMPT = (
    "You are a strict blind judge. Prompt given to both images:\n\"{prompt}\"\n\n"
    "Image A and Image B were generated from the same prompt. Which image "
    "matches the prompt better (object correctness, count, spatial relations, "
    "rendered text, overall quality)? Answer with exactly one of: A, B, TIE."
)


def parse_answer(text: str) -> str:
    t = text.strip().upper()
    for k in ("TIE", "A", "B"):
        if re.search(rf"\b{k}\b", t):
            return k
    return "UNPARSED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="population")
    ap.add_argument("--suite", default="alignment_100")
    ap.add_argument("--quant-arms", default="C_INT8,C_W4A8")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--swap-check", type=int, default=10)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image

    device = "cuda"
    snap = C.model_snapshot_path()
    processor = AutoProcessor.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(snap, "text_encoder"), torch_dtype=torch.float16, device_map=device,
        local_files_only=True)
    model.eval()

    prompts = {p["id"]: p for p in C.load_prompts(args.suite)}
    rng = random.Random(20261001)
    rows = []
    imgdir = os.path.join(C.VALDATA, "images", args.group)
    pids = [p["id"] for p in C.load_prompts(args.suite)]
    n_done = 0

    @torch.no_grad()
    def judge(prompt_text, img_a_path, img_b_path):
        qa = JUDGE_PROMPT.format(prompt=prompt_text)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img_a_path},
            {"type": "image", "image": img_b_path},
            {"type": "text", "text": qa}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]

    for pid in pids:
        for steps in (25, 40):
            ref_f = os.path.join(imgdir, f"{pid}_{steps}s_C_BF16.png")
            if not os.path.exists(ref_f):
                continue
            for qa in args.quant_arms.split(","):
                qf = os.path.join(imgdir, f"{pid}_{steps}s_{qa}.png")
                if not os.path.exists(qf):
                    continue
                if n_done >= args.limit:
                    break
                flip = rng.random() < 0.5
                a, b = (qf, ref_f) if flip else (ref_f, qf)
                raw = judge(prompts[pid]["text"], a, b)
                ans = parse_answer(raw)
                winner = {"A": qa if flip else "C_BF16",
                          "B": "C_BF16" if flip else qa,
                          "TIE": "TIE", "UNPARSED": "UNPARSED"}[ans]
                row = {"pid": pid, "steps": steps, "vs": qa, "flip": int(flip),
                       "raw": raw.strip()[:16], "parsed": ans, "winner": winner}
                # order-swap consistency subset
                if args.swap_check > 0:
                    raw2 = judge(prompts[pid]["text"], b, a)
                    ans2 = parse_answer(raw2)
                    row["swap_parsed"] = ans2
                    row["swap_consistent"] = int(
                        (ans == "TIE" and ans2 == "TIE") or
                        (ans == "A" and ans2 == "B") or (ans == "B" and ans2 == "A"))
                    args.swap_check -= 1
                rows.append(row)
                n_done += 1
                print(row, flush=True)

    mdir = os.path.join(C.REPO, "artifacts", "v2", "metrics")
    with open(os.path.join(mdir, f"vlm_judge_{args.group}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for qa in args.quant_arms.split(","):
        sub = [r for r in rows if r["vs"] == qa]
        if sub:
            summary[qa] = {
                "n": len(sub),
                "bf16_wins": sum(r["winner"] == "C_BF16" for r in sub),
                "quant_wins": sum(r["winner"] == qa for r in sub),
                "ties": sum(r["winner"] == "TIE" for r in sub),
                "unparsed": sum(r["winner"] == "UNPARSED" for r in sub),
                "swap_consistent_rate": (sum(r.get("swap_consistent", 1) for r in sub) /
                                          max(1, sum("swap_consistent" in r for r in sub)))}
    C.save_json({"judge_model": "Qwen3-VL-8B (Qwen-Image-2.1 text_encoder, official snapshot)",
                 "judge_prompt": JUDGE_PROMPT, "randomization": "seeded rng 20261001, recorded per row",
                 "summary": summary},
                os.path.join(mdir, f"vlm_judge_summary_{args.group}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
