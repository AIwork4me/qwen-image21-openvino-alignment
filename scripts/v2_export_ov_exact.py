#!/usr/bin/env python3
"""Build OpenVINO graphs that reproduce ComfyUI INT8 ConvRot / W4A8 linear
semantics EXACTLY (arm candidates O_INT8_EXACT / O_W4A8_EXACT).

Approach: load the official transformers Qwen3VL language model (fp32), then
replace every decoder Linear and the embedding with modules implementing the
exact comfy_kitchen eager math:

  rotation:      x_rot = (x.reshape(-1, K/256, 256) @ H)      (fp32, H = Hadamard-256)
  act quant:     xs = clamp(absmax(x_rot)/127, 1e-30); q = round_half_even(x_rot/xs) clamp [-128,127]
  GEMM:          acc = q.int32 @ Wq.int32.T                   (integer-exact)
  rescale:       y = acc.float() * (xs * scale_per_channel)

W4A8 uses Wq = decoded int8 grid (codebook[s_rel] rounded, clamped) with
scale = s_channel; INT8 uses artifact Wq with scale = per-channel weight_scale.
The embedding table is dequantized offline with the same fp32 ops (q*scale @ H)
and gathered at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

from safetensors import safe_open
import openvino as ov

from comfy_kitchen.backends.eager import w4a8_int8 as w4a8_eager
from comfy_kitchen.tensor.int8_utils import _build_hadamard


def hadamard256() -> torch.Tensor:
    return _build_hadamard(256, device="cpu", dtype=torch.float32)


class ExactInt8Linear(nn.Module):
    """ComfyUI int8_linear exact semantics (rotation + dynamic rowwise A8 + int32 GEMM)."""

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor, h: torch.Tensor):
        super().__init__()
        self.register_buffer("wq_i32", wq.to(torch.int32), persistent=False)
        self.register_buffer("scale", scale.float(), persistent=False)
        self.register_buffer("h", h, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = x.shape[-1]
        xr = x.reshape(-1, K // 256, 256) @ self.h
        xr = xr.reshape(x.shape)
        absmax = xr.abs().amax(dim=-1, keepdim=True)
        xs = (absmax / 127.0).clamp(min=1e-30)
        s4m = torch.where(xs == 0, torch.full_like(xs, torch.finfo(torch.float32).tiny), xs)
        q = (xr / s4m).round().clamp(-128.0, 127.0).to(torch.int32)
        acc = q @ self.wq_i32.T
        return acc.to(torch.float32) * (xs * self.scale)


def decode_w4a8_to_int8(qdata, s_rel, s_channel, codebook, group_size, K) -> torch.Tensor:
    from comfy_kitchen.backends.eager.w4a8_int8 import _unpack_codes
    N = qdata.shape[0]
    codes = _unpack_codes(qdata, K, 4)
    values = codebook.to(torch.float32)[codes]
    groups = K // group_size
    values = values.view(N, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(N, K).round().clamp(-127, 127).to(torch.int8)


def build_replacement(arm: str, artifact_path: str, model) -> dict:
    """Create ExactInt8Linear weights for all 36x7 linears from the artifact."""
    h = hadamard256()
    lm = model.model.language_model
    n_layers = len(lm.layers)
    repl = {}
    with safe_open(artifact_path, framework="pt", device="cpu") as f:
        cfgs = {k[:-len(".comfy_quant")]: json.loads(f.get_tensor(k).numpy().tobytes())
                for k in f.keys() if k.endswith(".comfy_quant")}
        for li in range(n_layers):
            for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                mod = f"model.layers.{li}.{proj}"
                cfg = cfgs[mod]
                wq = f.get_tensor(mod + ".weight")
                if cfg["format"] == "int8_tensorwise":
                    scale = f.get_tensor(mod + ".weight_scale").reshape(-1)
                    eff = wq
                elif cfg["format"] == "asym_w4a8_int8":
                    s_rel = f.get_tensor(mod + ".weight_s_rel")
                    s_ch = f.get_tensor(mod + ".weight_s_channel")
                    cb = f.get_tensor(mod + ".weight_codebook")
                    K = s_rel.shape[1] * cfg["group_size"]
                    eff = decode_w4a8_to_int8(wq, s_rel, s_ch, cb, cfg["group_size"], K)
                    scale = s_ch
                else:
                    raise ValueError(cfg)
                repl[mod] = (eff, scale)
    return repl, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["O_INT8_EXACT", "O_W4A8_EXACT"])
    ap.add_argument("--seq-len", type=int, default=48)
    args = ap.parse_args()
    out_dir = f"/valdata/models/v2_ov_{args.arm.lower()}"
    os.makedirs(out_dir, exist_ok=True)

    artifact = C.ENCODER_FILES["C_INT8" if args.arm == "O_INT8_EXACT" else "C_W4A8"]
    artifact_path = os.path.join(C.ENCODER_DIR, artifact)

    from transformers import Qwen3VLForConditionalGeneration
    te_dir = os.path.join(C.model_snapshot_path(), "text_encoder")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, torch_dtype=torch.float32)
    model.eval()
    print(f"loaded fp32 scaffold in {time.time()-t0:.0f}s", flush=True)

    repl, h = build_replacement(args.arm, artifact_path, model)
    print(f"built {len(repl)} exact linears", flush=True)

    # replace modules
    lm = model.model.language_model
    for li, layer in enumerate(lm.layers):
        for attr in ("self_attn", "mlp"):
            sub = getattr(layer, attr)
            for name in list(sub._modules):
                if hasattr(getattr(sub, name), "weight") and getattr(sub, name).__class__.__name__ == "Linear":
                    key = f"model.layers.{li}.{attr}.{name}" if name.endswith("_proj") else None
                    if name.endswith("_proj") and key in repl:
                        wq, scale = repl[key]
                        setattr(sub, name, ExactInt8Linear(wq, scale, h))

    # verify no remaining plain Linears in decoder
    remaining = [n for n, m in lm.named_modules() if isinstance(m, nn.Linear)]
    print("remaining plain Linears:", len(remaining), remaining[:3], flush=True)

    # dequantized embedding table (exact artifact semantics) for the runtime gather
    with safe_open(artifact_path, framework="pt", device="cpu") as f:
        eq = f.get_tensor("model.embed_tokens.weight")
        escale = f.get_tensor("model.embed_tokens.weight_scale").float()
        table = (eq.float() * escale)
        table = (table.reshape(-1, 4096 // 256, 256) @ h).reshape(-1, 4096).contiguous()
    tpath = os.path.join(out_dir, "embed_table_dequant_fp32.npy")
    np.save(tpath, table.numpy())
    print(f"embed table dequantized -> {tpath} ({table.shape})", flush=True)

    # instrumented wrapper (same as O_FP32)
    class TextOnlyQwen3VLExact(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.lm = m

        def forward(self, inputs_embeds, attention_mask, position_ids):
            captured = {}

            def norm_hook(module, args, kwargs, output):
                captured["prenorm"] = args[0]
                return args[0]

            hh = self.lm.norm.register_forward_hook(norm_hook, with_kwargs=True)
            try:
                out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              position_ids=position_ids, output_hidden_states=True, use_cache=False)
            finally:
                hh.remove()
            prenorm = captured["prenorm"]
            postnorm = self.lm.norm(prenorm)
            return (*out.hidden_states, prenorm, postnorm)

    wrapper = TextOnlyQwen3VLExact(lm)
    b, n = 1, args.seq_len
    ex_embeds = torch.randn(b, n, 4096, dtype=torch.float32)
    ex_attn = torch.ones(b, n, dtype=torch.int64)
    ar = torch.arange(n, dtype=torch.int64)
    ex_pos = torch.stack([ar, ar, ar]).reshape(3, b, n)
    with torch.no_grad():
        outs = wrapper(ex_embeds, ex_attn, ex_pos)
    assert len(outs) == 39

    t0 = time.time()
    with torch.no_grad():
        om = ov.convert_model(wrapper, example_input=(ex_embeds, ex_attn, ex_pos),
                              input=[(b, ov.Dimension(1, 2048), 4096), (b, ov.Dimension(1, 2048)),
                                     (3, b, ov.Dimension(1, 2048))])
    print(f"converted in {time.time()-t0:.0f}s", flush=True)
    xml = os.path.join(out_dir, "language_model_exact.xml")
    ov.save_model(om, xml, compress_to_fp16=False)
    C.save_json({"source_artifact": artifact,
                 "artifact_sha256": C.sha256_file(artifact_path),
                 "semantics": "comfy_kitchen eager int8_linear exact (Hadamard-256 rotation, "
                              "dynamic rowwise A8 half-to-even, int32 GEMM, fp32 rescale)",
                 "embed_table": tpath},
                os.path.join(out_dir, "conversion_manifest.json"))
    print("saved", xml, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
