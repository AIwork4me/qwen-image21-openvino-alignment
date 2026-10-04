#!/usr/bin/env python3
"""Phase A: freeze prompt preprocessing exactly as QwenImage21Pipeline does.

For every prompt: template -> processor (left padding) -> input_ids/attention_mask.
Saves artifacts/inputs/<pid>.npz + a manifest with token-level details.
Confirms text-only inputs have no pixel_values / image_grid_thw.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, npz_save

SYS_PROMPT = "Comprehend and analyze the provided prompt."
T2I_TEMPLATE = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)


def build_processor(snapshot):
    from transformers import Qwen3VLProcessor
    return Qwen3VLProcessor.from_pretrained(os.path.join(snapshot, "processor"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--prompts-file", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    snap = cfg["model"]["snapshot_path"]
    proc = build_processor(snap)
    tok = proc.tokenizer

    # drop_idx exactly as the pipeline computes it (tokenized system message length)
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
    sys_tokens = proc.apply_chat_template(sys_message, tokenize=True, return_dict=False)
    drop_idx = len(sys_tokens[0])
    img_token_id = tok.encode("<|image_pad|>")[0]
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")

    pf = args.prompts_file or f"prompts/{args.suite}.json"
    prompts = json.load(open(os.path.join(ROOT, pf)))["prompts"]

    outdir = os.path.join(ROOT, "artifacts", "inputs")
    os.makedirs(outdir, exist_ok=True)
    manifest = {"suite": args.suite, "prompts_file": pf, "drop_idx": drop_idx,
                "img_token_id": img_token_id, "im_start_id": im_start_id,
                "template": T2I_TEMPLATE, "processor_dir": os.path.join(snap, "processor"),
                "entries": []}

    for p in prompts:
        pid, text = p["id"], p["text"]
        formatted = T2I_TEMPLATE.format(text)
        mi = proc(text=[formatted], padding=True, padding_side="left", return_tensors="pt")
        keys = list(mi.keys())
        input_ids = mi.input_ids
        attn = mi.attention_mask
        assert input_ids.shape == attn.shape

        ids = input_ids[0].tolist()
        second_im_start = [i for i, t in enumerate(ids) if t == im_start_id][1]
        toks = tok.convert_ids_to_tokens(ids)
        entry = {
            "pid": pid, "raw": text, "formatted": formatted, "category": p.get("category"),
            "seq_len": int(input_ids.shape[1]),
            "keys": keys,
            "pixel_values_present": "pixel_values" in keys,
            "image_grid_thw_present": "image_grid_thw" in keys,
            "mm_token_type_ids_present": "mm_token_type_ids" in keys,
            "second_im_start_index": second_im_start,
            "second_im_start_equals_drop_idx": second_im_start == drop_idx,
            "n_image_pad_tokens": int((input_ids[0] == img_token_id).sum()),
            "attention_all_ones": bool((attn == 1).all().item()),
            "first_tokens": toks[:drop_idx + 3],
            "last_tokens": toks[-6:],
        }
        manifest["entries"].append(entry)
        npz_save(os.path.join(outdir, f"{pid}.npz"),
                 input_ids=input_ids.numpy().astype(np.int64),
                 attention_mask=attn.numpy().astype(np.int64),
                 mm_token_type_ids=(mi.mm_token_type_ids.numpy().astype(np.int64) if hasattr(mi, "mm_token_type_ids") else np.zeros_like(input_ids.numpy())),
                 drop_idx=np.array(drop_idx, dtype=np.int64))
        print(f"[{pid}] len={entry['seq_len']} drop_idx_check={entry['second_im_start_equals_drop_idx']} keys={keys}")

    save_json(manifest, os.path.join(outdir, f"manifest_{args.suite}.json"))
    print("saved manifest:", os.path.join(outdir, f"manifest_{args.suite}.json"))


if __name__ == "__main__":
    main()
