#!/usr/bin/env python3
"""v2: run the OFFICIAL ComfyUI Qwen-Image-2.1 text-encoder artifacts on W7900/ROCm.

Uses the real ComfyUI loader code (comfy.utils.load_torch_file -> convert_old_quants
-> llama_detect -> qwen_image21.te + QwenImage21Tokenizer), exactly the chain
comfy.sd.load_clip() executes for CLIPType.QWEN_IMAGE, and the comfy-kitchen HIP
kernels for int8_tensorwise+convrot and asym_w4a8_int8 linears.

Captures per prompt:
  - llama-formatted text, token ids, attention mask (input level)
  - all 37 hidden states (embedding output + 36 blocks) [fp32]
  - pre-final-norm hidden state (THE conditioning source, matches transformers-4.57
    hidden_states[-1] semantics that Qwen-Image 2.1 is tuned to)
  - post-final-norm hidden state
  - the actual conditioning tensor returned by QwenImage21TEModel.encode_token_weights
    (after the system-turn drop) -- this is what the DiT consumes
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, tensor_hash

_DEVICE = "cuda"

COMFY_SRC = os.environ.get("V2_COMFYUI_SRC", "/valdata/comfyui-src")
ARTIFACT_DIR = os.environ.get("V2_ENCODER_DIR", "/valdata/v2models/text_encensors") \
    if False else os.environ.get("V2_ENCODER_DIR", "/valdata/v2models/text_encoders")

ARMS = {
    "C_BF16": "qwen3vl_8b_bf16.safetensors",
    "C_INT8": "qwen3vl_8b_int8_convrot.safetensors",
    "C_W4A8": "qwen3vl_8b_w4a8.safetensors",
}


def import_comfy():
    sys.path.insert(0, COMFY_SRC)
    os.chdir("/tmp/opencode")  # avoid picking up repo-local modules named comfy
    sys.argv = ["comfyui-standalone-runner"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.utils
    import comfy.text_encoders.qwen_image21
    import comfy.text_encoders.hunyuan_video
    return comfy


def load_comfy_te(arm, device="cuda", dtype=None):
    """Replicates comfy.sd.load_clip(..., CLIPType.QWEN_IMAGE) for a TE-only checkpoint.

    dtype=None -> use the checkpoint's own dtype (llama_detect). Passing
    dtype=torch.float32 (semantics probe, C_BF16 only) upcasts the unquantized
    bf16 checkpoint to fp32 compute to separate kernel-order noise from
    semantic differences vs the official fp32 reference.
    """
    comfy = import_comfy()
    path = os.path.join(ARTIFACT_DIR, ARMS[arm])
    t0 = time.time()
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
    sd = comfy.utils.state_dict_prefix_replace(
        sd, {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
    detect = comfy.text_encoders.hunyuan_video.llama_detect(sd)
    load_s = time.time() - t0
    if dtype is None:
        te_cls = comfy.text_encoders.qwen_image21.te(**detect)
        te_model = te_cls(device=device, model_options={})
    else:
        if detect.get("llama_quantization_metadata") is not None:
            raise SystemExit("--dtype fp32 probe is only valid for the unquantized C_BF16 arm")
        te_cls = comfy.text_encoders.qwen_image21.te()
        te_model = te_cls(device=device, dtype=dtype, model_options={})
    tok = comfy.text_encoders.qwen_image21.QwenImage21Tokenizer()
    missing, unexpected = te_model.load_sd(sd)
    n_quant = sum(1 for k in sd if k.endswith(".comfy_quant"))
    info = {
        "arm": arm, "path": path, "load_s": load_s,
        "dtype_llama": str(detect.get("dtype_llama")),
        "quant_metadata_layers": len((detect.get("llama_quantization_metadata") or {}).get("layers", {})),
        "n_comfy_quant_keys": n_quant,
        "missing_keys": len(missing) if missing else 0,
        "unexpected_keys": len(unexpected) if unexpected else 0,
        "missing_sample": list(missing)[:8] if missing else [],
        "unexpected_sample": list(unexpected)[:8] if unexpected else [],
    }
    return te_model, tok, info


def encode_with_capture(te_model, prompt, fast_kernels=False):
    """Tokenize + encode with read-only hooks for layer outputs; returns everything.

    Mirrors comfy.sd.CLIP.encode_from_tokens: sets execution_device, and (unless
    --fast-kernels) does NOT enter comfy.ops.use_quantized_matmul -- matching the
    stock ComfyUI conditioning path (dequantize-to-compute-dtype + regular matmul).
    """
    tok = _TOK_STATE["tok"]
    token_weight_pairs = tok.tokenize_with_weights(prompt)

    clip_model = getattr(te_model, te_model.clip)
    clip_model.set_clip_options({"execution_device": torch.device(_DEVICE)})
    transformer = clip_model.transformer  # Qwen3VL
    llama_model = transformer.model      # Llama2_
    captured = {}
    handles = []

    def layer_hook(i):
        def hook(mod, args, kwargs, output):
            captured[f"layer_{i:02d}"] = output[0].detach().float().cpu() \
                if isinstance(output, tuple) else output.detach().float().cpu()
        return hook

    def embed_hook(mod, args, kwargs, output):
        captured["embedding"] = output.detach().float().cpu()

    def prenorm_hook(mod, args, kwargs, output):
        captured["prenorm"] = args[0].detach().float().cpu()

    def postnorm_hook(mod, args, kwargs, output):
        captured["postnorm"] = output.detach().float().cpu()

    handles.append(llama_model.embed_tokens.register_forward_hook(embed_hook, with_kwargs=True))
    for i, layer in enumerate(llama_model.layers):
        handles.append(layer.register_forward_hook(layer_hook(i), with_kwargs=True))
    if llama_model.norm is not None:
        handles.append(llama_model.norm.register_forward_hook(prenorm_hook, with_kwargs=True))
        handles.append(llama_model.norm.register_forward_hook(postnorm_hook, with_kwargs=True))
    try:
        t0 = time.time()
        import comfy.model_management as mm
        ctxs = [torch.no_grad(), mm.cuda_device_context(torch.device(_DEVICE))]
        if fast_kernels:
            import comfy.ops
            ctxs.append(comfy.ops.use_quantized_matmul(clip_model, torch.device(_DEVICE)))
        with contextlib.ExitStack() as stack:
            for c in ctxs:
                stack.enter_context(c)
            out, pooled, extra = te_model.encode_token_weights(token_weight_pairs)
        enc_s = time.time() - t0
    finally:
        for h in handles:
            h.remove()

    # token-level detail
    key = next(iter(token_weight_pairs))
    toks = [t[0] if not isinstance(t[0], dict) else "<image>" for t in token_weight_pairs[key][0]]
    return {
        "token_ids": toks,
        "seq_len": len(toks),
        "cond": out.detach().float().cpu(),
        "attention_mask": extra.get("attention_mask"),
        "captured": captured,
        "encode_s": enc_s,
        "pooled": pooled,
    }


_TOK_STATE = {"tok": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--fast-kernels", action="store_true",
                    help="wrap encode in comfy.ops.use_quantized_matmul (NOT the stock "
                         "conditioning path; supplementary arm for A8-kernel effect)")
    ap.add_argument("--dtype", default=None, choices=[None, "fp32"],
                    help="compute-dtype override; 'fp32' is a semantics probe for C_BF16 only")
    ap.add_argument("--outdir-name", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    prompts = json.load(open(os.path.join(ROOT, "prompts", f"{args.suite}.json")))["prompts"]

    dtype = torch.float32 if args.dtype == "fp32" else None
    te_model, tok, info = load_comfy_te(args.arm, device=args.device, dtype=dtype)
    _TOK_STATE["tok"] = tok
    if args.device.startswith("cuda"):
        info["vram_alloc_gib_after_load"] = torch.cuda.memory_allocated() / 2 ** 30
        info["vram_reserved_gib_after_load"] = torch.cuda.memory_reserved() / 2 ** 30
    print(f"[{args.arm}] loaded in {info['load_s']:.1f}s "
          f"(quant layers={info['quant_metadata_layers']}, missing={info['missing_keys']}, "
          f"unexpected={info['unexpected_keys']})")

    out_name = args.outdir_name or args.arm
    if args.dtype == "fp32":
        out_name += "_fp32compute"
    if args.fast_kernels:
        out_name += "_fastk"
    outdir = os.path.join(ROOT, "artifacts", "v2", "tensors", out_name)
    os.makedirs(outdir, exist_ok=True)

    entries = []
    for p in prompts:
        pid = p["id"]
        for rep in range(args.repeats):
            res = encode_with_capture(te_model, p["text"], fast_kernels=args.fast_kernels)
            rep_sfx = "" if args.repeats == 1 else f"_r{rep}"
            npz_path = os.path.join(outdir, f"{pid}{rep_sfx}.npz")
            arrays = {"cond": res["cond"].numpy()}
            for k in ("embedding", "prenorm", "postnorm"):
                if k in res["captured"]:
                    arrays[k] = res["captured"][k].numpy()
            for k, v in res["captured"].items():
                if k.startswith("layer_"):
                    arrays[k] = v.numpy()
            np.savez_compressed(npz_path, **arrays)
            entry = {
                "pid": pid, "repeat": rep, "text": p["text"],
                "seq_len": res["seq_len"], "token_ids": res["token_ids"],
                "cond_shape": list(res["cond"].shape),
                "cond_sha256": tensor_hash(res["cond"]),
                "prenorm_sha256": tensor_hash(res["captured"]["prenorm"]),
                "encode_s": res["encode_s"], "npz": npz_path,
            }
            entries.append(entry)
            print(f"[{args.arm}][{pid}{rep_sfx}] seq={res['seq_len']} "
                  f"cond={tuple(res['cond'].shape)} t={res['encode_s']:.2f}s")

    meta = {**info, "suite": args.suite, "repeats": args.repeats, "entries": entries,
            "comfyui_src": COMFY_SRC, "fast_kernels": bool(args.fast_kernels),
            "encode_path": "use_quantized_matmul (fast A8 kernels)" if args.fast_kernels
            else "stock encode_from_tokens path (dequantize-to-dtype + regular matmul)"}
    save_json(meta, os.path.join(outdir, f"meta_{out_name}_{args.suite}.json"))
    print("done", args.arm)


if __name__ == "__main__":
    main()
