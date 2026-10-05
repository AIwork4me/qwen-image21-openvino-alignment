#!/usr/bin/env python3
"""Run a v2 encoder arm over a prompt suite and freeze conditioning + layer trace.

Arms:
  R_FP32 / R_BF16  official Qwen3-VL-8B (transformers) on W7900, official
                   pipeline semantics (vendored QwenImage21Pipeline._get_qwen_prompt_embeds)
  C_BF16 / C_INT8 / C_W4A8  official ComfyUI artifacts loaded through
                   comfy.sd.load_text_encoder_state_dicts (CLIPType.QWEN_IMAGE) on W7900

Per (arm, prompt) saves /valdata/tensors/{arm}/{pid}.npz:
  input_ids, attention_mask, cond [T',4096] fp32, encoder mask,
  hidden_states [37,T,4096] fp32 (embed + 36 layer outputs, prenorm at -1),
  postnorm [T,4096] fp32, meta (hashes, config)
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

SNAP = C.model_snapshot_path()


# ---------------------------------------------------------------- R arms
def run_reference_arm(arm: str, suite: str, pids: list[str] | None, repeats: int = 1) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

    dtype = torch.float32 if arm == "R_FP32" else torch.bfloat16
    device = "cuda"
    processor = AutoProcessor.from_pretrained(os.path.join(SNAP, "processor"), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(SNAP, "text_encoder"), torch_dtype=dtype, device_map=device, local_files_only=True)
    model.eval()

    sys_prompt = "Comprehend and analyze the provided prompt."
    template = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n")
    sys_tokens = processor.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}],
        tokenize=True, return_dict=False)
    drop_idx = len(sys_tokens[0])
    text_model = getattr(model.model, "language_model", model.model)
    # replicate official norm-hook: capture prenorm input to final RMSNorm
    captured = {}

    def norm_hook(module, args, output):
        captured["prenorm"] = args[0].detach().clone()
    handle = text_model.norm.register_forward_hook(norm_hook)

    prompts = [p for p in C.load_prompts(suite) if pids is None or p["id"] in pids]
    outdir = os.path.join(C.VALDATA, "tensors", arm)
    os.makedirs(outdir, exist_ok=True)
    for rep in range(repeats):
        suffix = "" if repeats == 1 else f"_rep{rep}"
        for p in prompts:
            t0 = time.time()
            text = (p.get("text") or p.get("prompt") or " ")
            full = template.format(text)
            mi = processor(text=[full], padding=True, padding_side="left", return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(input_ids=mi.input_ids, attention_mask=mi.attention_mask,
                            output_hidden_states=True, return_dict=True)
            hs = [h.detach() for h in out.hidden_states]  # 37 entries: embed + 36 layers
            prenorm = captured["prenorm"].detach()
            # transformers 5.18 ties hidden_states[-1] to the NORMED last_hidden_state;
            # official semantics (pipeline hook) use the prenorm last-layer output:
            hs[-1] = prenorm
            postnorm = text_model.norm(prenorm).detach()
            # official conditioning: mask select, drop system tokens, right pad
            bool_mask = mi.attention_mask.bool()
            valid = bool_mask[0].sum().item()
            emb = prenorm[0][bool_mask[0]][drop_idx:]
            cond = emb.unsqueeze(0)
            mask_out = torch.ones(1, cond.shape[1], dtype=torch.long)
            ids = mi.input_ids[0][bool_mask[0]].cpu().numpy()
            np.savez_compressed(
                os.path.join(outdir, f"{p['id']}{suffix}.npz"),
                input_ids=ids,
                attention_mask=mi.attention_mask.cpu().numpy(),
                cond=cond.float().cpu().numpy(),
                encoder_mask=mask_out.numpy(),
                hidden_states=torch.stack([h[0][bool_mask[0]].float().cpu() for h in hs]).numpy(),
                postnorm=postnorm[0][bool_mask[0]].float().cpu().numpy(),
                meta=json.dumps({"arm": arm, "pid": p["id"], "drop_idx": drop_idx,
                                 "dtype": str(dtype), "seq_len": int(valid),
                                 "cond_sha256": C.tensor_sha256(cond.float().cpu()),
                                 "input_ids_sha256": C.tensor_sha256(mi.input_ids.cpu())}),
            )
            print(f"{arm} {p['id']}{suffix}: T={valid} drop={drop_idx} "
                  f"T'={cond.shape[1]} {time.time()-t0:.1f}s", flush=True)
    handle.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C arms
def load_comfy_clip(artifact: str):
    sys.path.insert(0, C.COMFYUI_ROOT)
    argv_backup = sys.argv
    sys.argv = [argv_backup[0]]
    try:
        import comfy.sd  # noqa: F401  (initializes folder_paths/model_management)
    finally:
        sys.argv = argv_backup
    import comfy.sd as sd
    import comfy.utils
    from safetensors.torch import load_file

    path = os.path.join(C.ENCODER_DIR, artifact)
    raw = load_file(path, device="cpu")  # no prefix replace: detect_te_model needs original keys
    clip = sd.load_text_encoder_state_dicts(
        [raw], embedding_directory=None, clip_type=sd.CLIPType.QWEN_IMAGE,
        model_options={"initial_device": "cpu"})
    return clip


def run_comfy_arm(arm: str, suite: str, pids: list[str] | None, repeats: int = 1) -> None:
    clip = load_comfy_clip(C.ENCODER_FILES[arm])
    tokenizr = clip.tokenizer
    cond_model = clip.cond_stage_model
    inner = cond_model.qwen3vl_8b  # QwenImage21Qwen3VLClipModel (SD1ClipModel attr keyed by name)
    transformer = inner.transformer  # Qwen3VL
    llama = transformer.model  # Llama2_
    layers = llama.layers

    captured = {"embed": None, "layers": []}

    def emb_hook(module, args, output):
        captured["embed"] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    handles = [llama.embed_tokens.register_forward_hook(emb_hook)]

    def layer_hook_factory(idx):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            captured["layers"].append(o.detach().clone())  # layer writes in-place into its input
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook_factory(i)))

    def norm_hook(module, args, output):
        captured["prenorm"] = args[0].detach().clone()
    handles.append(llama.norm.register_forward_hook(norm_hook))

    prompts = [p for p in C.load_prompts(suite) if pids is None or p["id"] in pids]
    outdir = os.path.join(C.VALDATA, "tensors", arm)
    os.makedirs(outdir, exist_ok=True)
    clip.load_model()  # move to GPU
    dev = clip.patcher.load_device

    for rep in range(repeats):
        suffix = "" if repeats == 1 else f"_rep{rep}"
        for p in prompts:
            t0 = time.time()
            tokens = tokenizr.tokenize_with_weights((p.get("text") or p.get("prompt") or " "))
            captured["layers"] = []
            cond, pooled, extra = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=False) \
                if False else clip.cond_stage_model.encode_token_weights(tokens)
            # replicate CLIP.encode_from_tokens plumbing: encode_token_weights returns
            # (cond, pooled, extra) directly for QwenImage21TEModel
            cond = cond.to(dev)
            ids_flat = [t[0] for t in tokens["qwen3vl_8b"][0]]
            ids = np.array([t for t in ids_flat if isinstance(t, int)], dtype=np.int64)
            n_layers = len(captured["layers"])
            hs = [captured["embed"]] + captured["layers"]
            hidden = torch.stack([h[0].float().cpu() for h in hs])
            prenorm = captured["prenorm"]
            postnorm = llama.norm(prenorm)
            enc_mask = extra.get("attention_mask", None)
            cond_out = cond[0].float().cpu()
            np.savez_compressed(
                os.path.join(outdir, f"{p['id']}{suffix}.npz"),
                input_ids=ids,
                attention_mask=(enc_mask[0].cpu().numpy() if enc_mask is not None
                                else np.ones(len(ids), dtype=np.int64)),
                cond=cond_out.numpy(),
                encoder_mask=(enc_mask[0].cpu().numpy() if enc_mask is not None
                              else np.ones(cond.shape[1], dtype=np.int64)),
                hidden_states=hidden.numpy(),
                postnorm=postnorm[0].float().cpu().numpy(),
                meta=json.dumps({"arm": arm, "pid": p["id"], "n_layer_captures": n_layers,
                                 "seq_len": int(len(ids)), "cond_T": int(cond.shape[1]),
                                 "cond_sha256": C.tensor_sha256(cond_out),
                                 "input_ids_sha256": C.tensor_sha256(torch.from_numpy(ids))}),
            )
            print(f"{arm} {p['id']}{suffix}: T={len(ids)} T'={cond.shape[1]} "
                  f"layers_captured={n_layers} {time.time()-t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    del clip, cond_model, transformer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(C.ARMS.keys()))
    ap.add_argument("--suite", default="canary")
    ap.add_argument("--pids", default=None, help="comma-separated prompt ids")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()
    if not args.skip_gate:
        C.gate_environment()
    pids = args.pids.split(",") if args.pids else None
    t0 = time.time()
    if args.arm in ("R_FP32", "R_BF16"):
        run_reference_arm(args.arm, args.suite, pids, args.repeats)
    elif args.arm in ("C_BF16", "C_INT8", "C_W4A8"):
        run_comfy_arm(args.arm, args.suite, pids, args.repeats)
    else:
        raise SystemExit(f"arm {args.arm} handled elsewhere (openvino)")
    print(f"TOTAL {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
