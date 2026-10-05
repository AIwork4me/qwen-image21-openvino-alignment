#!/usr/bin/env python3
"""v2 sensitive-component ablations (hybrid quantization recipes).

Base: transformers fp32 scaffold with the ComfyUI-quantized semantics
(ExactInt8Linear — proven weight- and embedding-equivalent to the official
artifacts in v2_weight_parity + encoder matrix).

Ablated components restored to unquantized official weights:
  INT8_e        INT8 + official embedding table
  INT8_fb       INT8 + BF16(final block 35)
  INT8_f2b      INT8 + BF16(blocks 34,35)
  INT8_efb      INT8 + BF16(embedding + final block)
  W4A8_*       same four variants on the W4A8 base

Saves /valdata/tensors/{arm}/{pid}.npz with the standard v2 contract, so the
arms plug directly into the transplant/population tooling.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C
import v2_run_encoder_arm as R
from v2_export_ov_exact import ExactInt8Linear, build_replacement, hadamard256

from safetensors import safe_open

ABL_ARMS = {
    "INT8_e":   ("C_INT8", embed=True,  final_blocks=0),
    "INT8_fb":  ("C_INT8", embed=False, final_blocks=1),
    "INT8_f2b": ("C_INT8", embed=False, final_blocks=2),
    "INT8_efb": ("C_INT8", embed=True,  final_blocks=1),
    "W4A8_e":   ("C_W4A8", embed=True,  final_blocks=0),
    "W4A8_fb":  ("C_W4A8", embed=False, final_blocks=1),
    "W4A8_f2b": ("C_W4A8", embed=False, final_blocks=2),
    "W4A8_efb": ("C_W4A8", embed=True,  final_blocks=1),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ABL_ARMS))
    ap.add_argument("--suite", default="canary")
    args = ap.parse_args()
    C.gate_environment()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    snap = C.model_snapshot_path()
    processor = AutoProcessor.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)
    sys_prompt = "Comprehend and analyze the provided prompt."
    template = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n")
    sys_tokens = processor.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}],
        tokenize=True, return_dict=False)
    drop_idx = len(sys_tokens[0])

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(snap, "text_encoder"), torch_dtype=torch.float32).eval()
    lm = model.model.language_model
    n_layers = len(lm.layers)
    prompts = C.load_prompts(args.suite)

    for arm in args.arms.split(","):
        base_artifact, abl_embed, abl_blocks = ABL_ARMS[arm]
        artifact_path = os.path.join(C.ENCODER_DIR, C.ENCODER_FILES[base_artifact])
        repl, h = build_replacement(base_artifact, artifact_path, model)
        skip = set(range(n_layers - abl_blocks, n_layers)) if abl_blocks else set()
        n_repl = 0
        for li, layer in enumerate(lm.layers):
            if li in skip:
                continue
            for attr in ("self_attn", "mlp"):
                sub = getattr(layer, attr)
                for name in list(sub._modules):
                    if name.endswith("_proj"):
                        key = f"model.layers.{li}.{attr}.{name}"
                        if key in repl:
                            wq, scale = repl[key]
                            setattr(sub, name, ExactInt8Linear(wq, scale, h))
                            n_repl += 1
        # embedding ablation: restore official fp32 table
        if abl_embed:
            pass  # never replaced: official table stays
        else:
            with safe_open(artifact_path, framework="pt", device="cpu") as f:
                eq = f.get_tensor("model.embed_tokens.weight")
                escale = f.get_tensor("model.embed_tokens.weight_scale").float()
                table = (eq.float() * escale)
                table = (table.reshape(-1, 4096 // 256, 256) @ h).reshape(-1, 4096).contiguous()
            with torch.no_grad():
                lm.embed_tokens.weight.copy_(table)

        outdir = os.path.join(C.VALDATA, "tensors", f"ABL_{arm}")
        os.makedirs(outdir, exist_ok=True)
        model = model.to("cuda")
        captured = {}

        def norm_hook(module, args, output):
            captured["prenorm"] = args[0].detach().clone()
        handle = lm.norm.register_forward_hook(norm_hook)

        for p in prompts:
            t0 = time.time()
            full = template.format(p.get("text") or " ")
            mi = processor(text=[full], padding=True, padding_side="left",
                           return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(input_ids=mi.input_ids, attention_mask=mi.attention_mask,
                            output_hidden_states=False, return_dict=True)
            bool_mask = mi.attention_mask.bool()
            prenorm = captured["prenorm"]
            emb = prenorm[0][bool_mask[0]][drop_idx:]
            cond = emb.unsqueeze(0)
            np.savez_compressed(
                os.path.join(outdir, f"{p['id']}.npz"),
                input_ids=mi.input_ids[0][bool_mask[0]].cpu().numpy(),
                attention_mask=mi.attention_mask.cpu().numpy(),
                cond=cond.float().cpu().numpy(),
                encoder_mask=np.ones(cond.shape[1], dtype=np.int64),
                meta=json.dumps({"arm": f"ABL_{arm}", "pid": p["id"], "base": base_artifact,
                                 "abl_embed": abl_embed, "abl_final_blocks": abl_blocks,
                                 "cond_sha256": C.tensor_sha256(cond.float().cpu())}))
            print(f"ABL_{arm} {p['id']}: T'={cond.shape[1]} {time.time()-t0:.1f}s", flush=True)
        handle.remove()
        model = model.to("cpu")
        torch.cuda.empty_cache()

        # restore scaffold: rebuild linears + embedding from checkpoint
        del repl
        gc.collect()
        from transformers import Qwen3VLForConditionalGeneration as Q
        del model
        gc.collect()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(snap, "text_encoder"), torch_dtype=torch.float32).eval()
        lm = model.model.language_model
    return 0


if __name__ == "__main__":
    sys.exit(main())
