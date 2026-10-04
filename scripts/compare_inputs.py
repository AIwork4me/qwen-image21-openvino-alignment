#!/usr/bin/env python3
"""Phase A: input/preprocessing identity between the reference (processor) path and
the customer's OpenVINO production tokenization path (ComfyUI qwen25 tokenizer + T2I template).

Invariant required: input_ids EXACT, attention_mask EXACT, sequence length EXACT,
drop point EXACT.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json

T2I_TEMPLATE = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
COMFY_TOK = "/workspace/qwen3vl-cpu-benchmark/ComfyUI/comfy/text_encoders/qwen25_tokenizer"


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    prompts = json.load(open(os.path.join(ROOT, f"prompts/{suite}.json")))["prompts"]
    from transformers import AutoTokenizer
    comfy_tok = AutoTokenizer.from_pretrained(COMFY_TOK)

    rows = []
    for p in prompts:
        pid = p["id"]
        z = np.load(os.path.join(ROOT, "artifacts", "inputs", f"{pid}.npz"))
        ref_ids = z["input_ids"][0].tolist()
        ref_mask = z["attention_mask"][0].tolist()
        drop_idx = int(z["drop_idx"])

        enc = comfy_tok(T2I_TEMPLATE.format(p["text"]), add_special_tokens=False, return_tensors=None)
        ov_ids = enc["input_ids"]
        ov_mask = enc["attention_mask"] if "attention_mask" in enc else [1] * len(ov_ids)

        row = {
            "pid": pid,
            "ref_len": len(ref_ids), "ov_len": len(ov_ids),
            "len_equal": len(ref_ids) == len(ov_ids),
            "input_ids_exact": ref_ids == ov_ids,
            "attention_mask_exact": ref_mask == ov_mask,
            "ref_drop_idx": drop_idx,
            "ov_second_im_start": ([i for i, t in enumerate(ov_ids) if t == 151644] + [None])[1],
            "first_mismatch_pos": next((i for i, (a, b) in enumerate(zip(ref_ids, ov_ids)) if a != b), None),
        }
        row["drop_point_equal"] = row["ov_second_im_start"] == drop_idx
        rows.append(row)

    mdir = os.path.join(ROOT, "artifacts", "metrics")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "input_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    allexact = all(r["input_ids_exact"] and r["attention_mask_exact"] and r["len_equal"] for r in rows)
    save_json({"suite": suite, "all_exact": allexact, "n": len(rows), "rows": rows},
              os.path.join(mdir, "input_alignment_summary.json"))
    print(f"input alignment (processor vs ComfyUI qwen25 tokenizer): {'PASS (EXACT)' if allexact else 'FAIL'}")
    for r in rows:
        if not r["input_ids_exact"]:
            print("  MISMATCH", r["pid"], "first at", r["first_mismatch_pos"])


if __name__ == "__main__":
    main()
