#!/usr/bin/env python3
"""v2 OpenVINO export: clean FP32 text-only Qwen3-VL wrapper (arm O_FP32 source).

Instrumented wrapper (37 hidden states + prenorm + postnorm), embedding done at
runtime as a bit-exact gather (see v2_run_ov_arm.py).  Dynamic seq [1, 2048].
"""
from __future__ import annotations

import argparse
import gc as _gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C

import openvino as ov


class TextOnlyQwen3VL(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.lm = m.model.language_model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        captured = {}

        def norm_hook(module, args, kwargs, output):
            captured["prenorm"] = args[0]
            return args[0]

        h = self.lm.norm.register_forward_hook(norm_hook, with_kwargs=True)
        try:
            out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          position_ids=position_ids, output_hidden_states=True, use_cache=False)
        finally:
            h.remove()
        prenorm = captured["prenorm"]
        postnorm = self.lm.norm(prenorm)
        return (*out.hidden_states, prenorm, postnorm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/valdata/models/v2_ov_fp32")
    ap.add_argument("--seq-len", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import Qwen3VLForConditionalGeneration
    te_dir = os.path.join(C.model_snapshot_path(), "text_encoder")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, torch_dtype=torch.float32)
    model.eval()
    print(f"loaded fp32 in {time.time()-t0:.0f}s", flush=True)

    wrapper = TextOnlyQwen3VL(model)
    b, n = 1, args.seq_len
    ex_embeds = torch.randn(b, n, 4096, dtype=torch.float32)
    ex_attn = torch.ones(b, n, dtype=torch.int64)
    ar = torch.arange(n, dtype=torch.int64)
    ex_pos = torch.stack([ar, ar, ar]).reshape(3, b, n)

    with torch.no_grad():
        outs = wrapper(ex_embeds, ex_attn, ex_pos)
    assert len(outs) == 39, len(outs)
    print("eager outputs:", len(outs), "hs[36]==prenorm bitwise:",
          torch.equal(outs[36], outs[37]))
    del model, wrapper
    _gc.collect()

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
    print(f"converted in {time.time()-t0:.0f}s", flush=True)
    xml = os.path.join(args.out, "language_model_fp32.xml")
    ov.save_model(lm_ov, xml, compress_to_fp16=False)
    C.save_json({"source": te_dir, "tool": f"openvino {ov.__version__} pytorch-frontend convert_model",
                 "dtype": "fp32", "compress_to_fp16": False,
                 "outputs": "37 hidden states + prenorm + postnorm",
                 "seq_dim": "dynamic [1,2048]"},
                os.path.join(args.out, "conversion_manifest.json"))
    print("saved", xml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
