#!/usr/bin/env python3
"""v2: commit profiler evidence for the stock ComfyUI conditioning path.

Runs torch.profiler over one C_INT8 and one C_W4A8 encode (stock path) and one
fast-kernel encode, and records:
  - dequantize kernel call counts (per encode)
  - int8 GEMM kernel call counts
  - attention kernel families (CPU vs GPU)
Outputs artifacts/v2/metrics/stock_path_profile.json + fastk for the specs.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/valdata/comfyui-src")
os.chdir("/tmp/opencode")
sys.argv = ["profiler"]
import comfy.options
comfy.options.enable_args_parsing()

from run_comfy_encoder import load_comfy_te, _TOK_STATE, _DEVICE
from common import ROOT, save_json


def profile_encode(arm, fast_kernels):
    import comfy.model_management as mm
    import comfy.ops
    te_model, tok, info = load_comfy_te(arm, device=_DEVICE)
    _TOK_STATE["tok"] = tok
    twp = tok.tokenize_with_weights("A white coffee mug with the text \"GOOD MORNING\" printed in bold black letters.")
    clip_model = getattr(te_model, te_model.clip)
    clip_model.set_clip_options({"execution_device": torch.device(_DEVICE)})

    def run():
        import comfy.ops
        ctxs = [torch.no_grad()]
        if fast_kernels:
            ctxs.append(comfy.ops.use_quantized_matmul(clip_model, torch.device(_DEVICE)))
        import contextlib
        with contextlib.ExitStack() as stack:
            for c in ctxs:
                stack.enter_context(c)
            te_model.encode_token_weights(twp)
        torch.cuda.synchronize()

    run()  # warmup (lazy load)

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
    evs = prof.key_averages()
    report = []
    for e in evs:
        if e.self_cpu_time_total < 10_000_000 and e.self_device_time_total < 1_000_000:
            if not any(s in e.key for s in ("dequant", "int8_linear", "w4a8", "mm", "sdpa",
                                            "attention", "embedding")):
                continue
        report.append({"op": e.key[:100], "count": e.count,
                       "self_cpu_s": e.self_cpu_time_total / 1e6,
                       "self_cuda_s": e.self_device_time_total / 1e6})
    counts = {
        "dequantize_int8_convrot_weight_dtype": sum(e["count"] for e in report
                                                    if "dequantize_int8_convrot" in e["op"]),
        "dequantize_w4a8_int8_weight": sum(e["count"] for e in report
                                           if "dequantize_w4a8_int8" in e["op"]),
        "dequantize_int8_embedding": sum(e["count"] for e in report
                                         if "dequantize_int8_embedding" in e["op"]),
        "dequant_grouped_to_int8_kernel_w4": sum(e["count"] for e in report
                                                 if "dequant_grouped_to_int8_kernel" in e["op"]),
        "int8_linear_kernel": sum(e["count"] for e in report if e["op"].startswith("comfy_kitchen::int8_linear")),
        "w4a8_int8_linear_kernel": sum(e["count"] for e in report if e["op"].startswith("comfy_kitchen::w4a8_int8_linear")),
        "cpu_attention": sum(e["count"] for e in report if "for_cpu" in e["op"]),
        "aten_mm": sum(e["count"] for e in report if e["op"] == "aten::mm"),
    }
    del te_model
    torch.cuda.empty_cache()
    return counts, report


def main():
    out = {}
    for arm in ("C_INT8", "C_W4A8"):
        counts, detail = profile_encode(arm, fast_kernels=False)
        out[f"{arm}_stock"] = counts
        print(arm, "stock:", counts, flush=True)
        save_json({"detail": detail}, os.path.join(ROOT, "artifacts", "v2", "metrics",
                                                    f"stock_path_profile_{arm}.json"))
    counts, detail = profile_encode("C_INT8", fast_kernels=True)
    out["C_INT8_fastk"] = counts
    print("C_INT8 fastk:", counts, flush=True)
    save_json({"detail": detail}, os.path.join(ROOT, "artifacts", "v2", "metrics",
                                                "fastk_path_profile_C_INT8.json"))
    save_json(out, os.path.join(ROOT, "artifacts", "v2", "metrics", "stock_path_profile.json"))


if __name__ == "__main__":
    main()
