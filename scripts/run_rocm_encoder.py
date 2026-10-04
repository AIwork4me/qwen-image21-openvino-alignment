#!/usr/bin/env python3
"""Run the PyTorch/ROCm Qwen3-VL-8B text encoder (arms R0/R0R/R1) and freeze all outputs.

Captures per prompt:
  - all 37 hidden_states (embeddings + 36 layers) [fp32]
  - pre-final-norm last-layer hidden (THE conditioning tensor, via norm input hook)
  - post-final-norm last hidden
  - prompt_embeds after pipeline semantics (mask-extract -> drop_idx -> bf16 cast view)
Saves .npz (fp32) + .pt (raw dtype) + meta with timings and hash chain.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, npz_save, save_json, tensor_hash


def load_encoder(precision, snapshot):
    from transformers import Qwen3VLForConditionalGeneration
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    te_dir = os.path.join(snapshot, "text_encoder")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, torch_dtype=dtype)
    model = model.to("cuda")
    model.eval()
    load_s = time.time() - t0
    text_model = getattr(model.model, "language_model", model.model)
    return model, text_model, load_s


def encode_prompt(model, text_model, input_ids, attn, mm_token_type_ids):
    """Mirror QwenImage21Pipeline._get_qwen_prompt_embeds exactly (hook neutralizes final norm)."""
    captured = {}

    def capture_norm_input(module, args, kwargs, output):
        captured["prenorm"] = args[0].detach()
        return args[0]  # replace output with input (pipeline semantics)

    handle = text_model.norm.register_forward_hook(capture_norm_input, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True,
                        **({"mm_token_type_ids": mm_token_type_ids} if mm_token_type_ids is not None else {}))
    finally:
        handle.remove()
    return out, captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="R0 / R0R / R1")
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--precision", default=None, help="override: bf16|fp32")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_cfg()
    arm_cfg = cfg["encoder_arms"][args.arm]
    precision = args.precision or arm_cfg["precision"]
    snap = cfg["model"]["snapshot_path"]

    torch.backends.cudnn.benchmark = False
    meta_arm = {"arm": args.arm, "precision": precision, "impl": "pytorch_rocm",
                "torch": torch.__version__, "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "repeats": args.repeats}

    model, text_model, load_s = load_encoder(precision, snap)
    meta_arm["model_load_s"] = load_s
    meta_arm["vram_alloc_gib_after_load"] = torch.cuda.memory_allocated() / 2 ** 30
    print(f"[{args.arm}] loaded in {load_s:.1f}s, VRAM alloc {meta_arm['vram_alloc_gib_after_load']:.2f} GiB")

    prompts = json.load(open(os.path.join(ROOT, f"prompts/{args.suite}.json")))["prompts"]
    outdir = os.path.join(ROOT, "artifacts", "tensors", args.arm)
    os.makedirs(outdir, exist_ok=True)

    entries = []
    for rep in range(args.repeats):
        for p in prompts:
            pid = p["id"] if rep == 0 else f"{p['id']}_rep{rep}"
            z = np.load(os.path.join(ROOT, "artifacts", "inputs", f"{p['id']}.npz"))
            input_ids = torch.from_numpy(z["input_ids"]).to(args.device)
            attn = torch.from_numpy(z["attention_mask"]).to(args.device)
            mm = torch.from_numpy(z["mm_token_type_ids"]).to(args.device) if "mm_token_type_ids" in z else torch.zeros_like(input_ids)

            torch.cuda.synchronize()
            t0 = time.time()
            out, captured = encode_prompt(model, text_model, input_ids, attn, mm)
            torch.cuda.synchronize()
            t = time.time() - t0

            hs = torch.stack([h[0].float() for h in out.hidden_states]).cpu().numpy()  # [L+1, seq, D]
            prenorm = captured["prenorm"][0].float().cpu().numpy()
            with torch.no_grad():
                postnorm = text_model.norm(captured["prenorm"].detach())[0].float().cpu().numpy()
            assert hs.shape[0] == 37, hs.shape

            drop_idx = int(z["drop_idx"])
            # pipeline semantics: extract valid tokens, drop system prefix; batch=1 & all-ones mask
            prompt_embeds = prenorm[drop_idx:]
            pe_bf16_view = torch.from_numpy(prompt_embeds).to(torch.bfloat16).float().numpy()

            npz_save(os.path.join(outdir, f"{pid}.npz"),
                     hidden_states=hs, prenorm=prenorm, postnorm=postnorm,
                     prompt_embeds=prompt_embeds, prompt_embeds_bf16_view=pe_bf16_view,
                     drop_idx=np.array(drop_idx))
            torch.save({"prenorm_raw": captured["prenorm"][0].cpu(),
                        "prompt_embeds_raw": captured["prenorm"][0][drop_idx:].cpu()},
                       os.path.join(outdir, f"{pid}_rawdtype.pt"))
            entries.append({"pid": pid, "base_pid": p["id"], "repeat": rep, "seq_len": int(hs.shape[1]),
                            "kept_len": int(prompt_embeds.shape[0]), "encode_s": t,
                            "prenorm_hash": tensor_hash(prenorm), "pe_hash": tensor_hash(prompt_embeds)})
            print(f"[{args.arm}][{pid}] seq={hs.shape[1]} kept={prompt_embeds.shape[0]} t={t:.3f}s")

    meta_arm["entries"] = entries
    save_json(meta_arm, os.path.join(outdir, f"meta_{args.arm}_{args.suite}.json"))
    del model
    torch.cuda.empty_cache()
    print("done", args.arm)


if __name__ == "__main__":
    main()
