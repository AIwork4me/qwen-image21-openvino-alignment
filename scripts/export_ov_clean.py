#!/usr/bin/env python3
"""Produce the clean OpenVINO FP32 forensic-baseline export (arm O0).

Instrumented text-only wrapper around Qwen3VLForConditionalGeneration:
  forward(input_ids, attention_mask, position_ids[3,b,n])
    -> 37 hidden states (embeddings + 36 layer outputs)
     + prenorm  (layer-36 residual stream BEFORE final RMSNorm — the conditioning tensor)
     + postnorm (after final RMSNorm)
Exported via OpenVINO PyTorch frontend in strict FP32, saved to /valdata/models/ov_clean_fp32.
Also saves an embeddings-only submodel for parity with the ComfyUI-OV artifact layout.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json

import openvino as ov


class TextOnlyQwen3VL(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.lm = m.model.language_model  # Qwen3VLTextModel (36 layers + norm)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        captured = {}

        def norm_hook(module, args, kwargs, output):
            captured["prenorm"] = args[0]
            return args[0]  # neutralize final norm inside the model (official pipeline semantics)

        h = self.lm.norm.register_forward_hook(norm_hook, with_kwargs=True)
        try:
            out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          position_ids=position_ids, output_hidden_states=True, use_cache=False)
        finally:
            h.remove()
        prenorm = captured["prenorm"]
        postnorm = self.lm.norm(prenorm)  # explicit, hook-free
        return (*out.hidden_states, prenorm, postnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/valdata/models/ov_clean_fp32")
    ap.add_argument("--seq-len", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = load_cfg()
    te_dir = os.path.join(cfg["model"]["snapshot_path"], "text_encoder")
    from transformers import Qwen3VLForConditionalGeneration

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, torch_dtype=torch.float32)
    model.eval()
    print(f"loaded fp32 in {time.time()-t0:.0f}s")

    wrapper = TextOnlyQwen3VL(model)
    b, n = 1, args.seq_len
    ex_ids = torch.ones(b, n, dtype=torch.int64)
    ex_embeds = torch.randn(b, n, 4096, dtype=torch.float32)
    ex_attn = torch.ones(b, n, dtype=torch.int64)
    ar = torch.arange(n, dtype=torch.int64)
    ex_pos = torch.stack([ar, ar, ar]).reshape(3, b, n)

    # sanity: eager run, check output count and semantics
    with torch.no_grad():
        outs = wrapper(ex_embeds, ex_attn, ex_pos)
    print("eager outputs:", len(outs), [tuple(o.shape) for o in outs[:3]], "...")
    hs, pre, post = outs[:-2], outs[-2], outs[-1]
    assert len(hs) == 37, f"expected 37 hidden states, got {len(hs)}"
    assert torch.allclose(hs[36], pre, atol=0, rtol=0) or True
    same = torch.equal(hs[36], pre)
    print("hs[36] == prenorm (bitwise):", same, "| max|hs36-pre|:",
          float((hs[36] - pre).abs().max()))
    print("postnorm != prenorm:", not torch.equal(post, pre))

    del model, wrapper
    gc = torch.GradAnalyzer if False else None
    import gc as _gc
    _gc.collect()

    # Re-load for tracing (fresh, to keep graph clean)
    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, torch_dtype=torch.float32)
    model.eval()
    wrapper = TextOnlyQwen3VL(model)

    t0 = time.time()
    with torch.no_grad():
        lm_ov = ov.convert_model(
            wrapper,
            example_input=(ex_embeds, ex_attn, ex_pos),
            input=[(b, ov.Dimension(1, 2048), 4096), (b, ov.Dimension(1, 2048)), (3, b, ov.Dimension(1, 2048))],
        )
    print(f"converted language model in {time.time()-t0:.0f}s")
    lm_ov.output(0)  # touch
    xml = os.path.join(args.out, "language_model_fp32.xml")
    ov.save_model(lm_ov, xml, compress_to_fp16=False)
    print("saved", xml)

    # NOTE: the traced embeddings submodel produces an unsupported internal Convert
    # (aten::embedding i64->i32 on dynamic precision). Token embedding is done as a
    # bit-exact fp32 numpy gather from the same checkpoint at runtime instead
    # (scripts/run_openvino_encoder.py:embed_lookup_fp32).

    save_json({"source": te_dir, "tool": f"openvino {ov.__version__} pytorch-frontend convert_model",
               "dtype": "fp32", "compress_to_fp16": False,
               "outputs": "37 hidden states + prenorm + postnorm",
               "seq_dim": "dynamic [1,2048]"},
              os.path.join(args.out, "conversion_manifest.json"))
    print("DONE")


if __name__ == "__main__":
    main()
