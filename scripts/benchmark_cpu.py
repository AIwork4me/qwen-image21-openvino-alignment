#!/usr/bin/env python3
"""Section 23: CPU performance measurements for the encoder arms.

Measures (separate from ALL accuracy conclusions):
  model load time (read+compile), cold first encode, warm encode latency
  (median/p90/p95 over N iters), peak RSS via resource monitor.
Arms: R0-on-CPU (PyTorch fp32 reference, if practical), O0 (FP32), O1 (BF16 runtime),
O3r (INT8 reproduced), O3 (customer artifact, <=40-token prompts only on this host).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json, ResourceMonitor


def bench_ov(arm, prompts_inputs, n_warm, n_iter, threads=None):
    import openvino as ov
    from run_openvino_encoder import (CUSTOMER_INT8, build_feed, embed_lookup_fp32,
                                      find_prenorm_anchor, layer_output_ops)
    core = ov.Core()
    art = {"O0": "/valdata/models/ov_clean_fp32", "O3r": "/valdata/models/ov_clean_int8",
           "O3qL": "/valdata/models/ov_clean_int8_acc_full", "O3": CUSTOMER_INT8}.get(arm)
    p = {"EXECUTION_MODE_HINT": "PERFORMANCE"}
    if arm == "O1":
        p["INFERENCE_PRECISION_HINT"] = "bf16"
        art = "/valdata/models/ov_clean_fp32"
    else:
        p["INFERENCE_PRECISION_HINT"] = "f32"
    if threads:
        p["INFERENCE_NUM_THREADS"] = threads
    t0 = time.perf_counter()
    if arm == "O3":
        lm = core.read_model(os.path.join(art, "openvino_language_model.xml"))
        pre_name, norm_name = find_prenorm_anchor(lm)
        lops = layer_output_ops(lm)
        pre_op = next(op for op in lm.get_ordered_ops() if op.get_friendly_name() == pre_name)
        lm.add_outputs([op.output(0) for op in lops] + [pre_op.output(0)])
    else:
        lm = core.read_model(os.path.join(art, "language_model_fp32.xml" if arm in ("O0", "O1") else "language_model_int8.xml"))
    read_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = core.compile_model(lm, "CPU", p)
    compile_s = time.perf_counter() - t0

    int8_tbl = None
    if arm in ("O3r",):
        tz = np.load("/valdata/models/ov_clean_int8/embed_table_int8.npz")
        int8_tbl = (tz["q"], tz["zero_point"], tz["scale"])
    emb_c = None
    if arm == "O3":
        emb_c = core.compile_model(core.read_model(os.path.join(art, "openvino_text_embeddings_model.xml")), "CPU", p)

    cfg = load_cfg()
    results = {"arm": arm, "read_s": read_s, "compile_s": compile_s,
               "load_s": read_s + compile_s, "props": {k: str(v) for k, v in p.items()},
               "prompts": {}}
    mon = ResourceMonitor(os.path.join(ROOT, "artifacts", "metrics", f"cpu_bench_{arm}_rss.csv"), hz=10).start()
    import resource as res_mod
    base_rss = res_mod.getrusage(res_mod.RUSAGE_SELF).ru_maxrss / 1024
    req = c.create_infer_request()
    for pid, ids in prompts_inputs:
        if arm == "O3":
            e = np.asarray(emb_c({0: ids})[0], dtype=np.float32)
        else:
            e = embed_lookup_fp32(cfg["model"]["snapshot_path"], ids, int8_table=int8_tbl)
        feed = build_feed(c.inputs, ids, e)
        req.reset_state()
        t0 = time.perf_counter()
        req.infer(feed)
        cold = time.perf_counter() - t0
        times = []
        for _ in range(n_warm + n_iter):
            req.reset_state()
            t0 = time.perf_counter()
            req.infer(feed)
            times.append(time.perf_counter() - t0)
        warm = times[n_warm:]
        results["prompts"][pid] = {
            "n_tokens": int(ids.shape[1]),
            "cold_s": cold,
            "warm_median_s": float(np.median(warm)),
            "warm_p90_s": float(np.percentile(warm, 90)),
            "warm_p95_s": float(np.percentile(warm, 95)),
            "warm_mean_s": float(np.mean(warm)),
            "emb_lookup_s_excluded": True,
        }
        print(f"[{arm}][{pid}] n={ids.shape[1]} cold={cold:.3f}s warm_p50={np.median(warm):.3f}s p95={np.percentile(warm,95):.3f}s")
    mon.stop()
    results["peak_rss_gib"] = res_mod.getrusage(res_mod.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    results["base_rss_gib_before_load"] = base_rss / 1024
    return results


def bench_torch_cpu(prompts_inputs, n_warm, n_iter):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    cfg = load_cfg()
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(cfg["model"]["snapshot_path"], "text_encoder"), torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(min(64, os.cpu_count()))
    load_s = time.perf_counter() - t0
    lm = model.model.language_model
    results = {"arm": "R0_cpu_fp32", "read_plus_load_s": load_s, "prompts": {}}
    import resource as res_mod
    for pid, ids in prompts_inputs:
        t = torch.from_numpy(ids)
        mm = torch.zeros_like(t)
        with torch.no_grad():
            def nh(m, a, k, o):
                return a[0]
            h = lm.norm.register_forward_hook(nh, with_kwargs=True)
            t0 = time.perf_counter()
            model(input_ids=t, attention_mask=torch.ones_like(t), mm_token_type_ids=mm,
                  output_hidden_states=True)
            cold = time.perf_counter() - t0
            h.remove()
            times = []
            for _ in range(n_warm + n_iter):
                h = lm.norm.register_forward_hook(nh, with_kwargs=True)
                t0 = time.perf_counter()
                model(input_ids=t, attention_mask=torch.ones_like(t), mm_token_type_ids=mm,
                      output_hidden_states=True)
                times.append(time.perf_counter() - t0)
                h.remove()
        warm = times[n_warm:]
        results["prompts"][pid] = {"n_tokens": int(ids.shape[1]), "cold_s": cold,
                                   "warm_median_s": float(np.median(warm)),
                                   "warm_p90_s": float(np.percentile(warm, 90)),
                                   "warm_p95_s": float(np.percentile(warm, 95))}
        print(f"[R0_cpu][{pid}] n={ids.shape[1]} cold={cold:.2f}s warm_p50={np.median(warm):.2f}s")
    results["peak_rss_gib"] = res_mod.getrusage(res_mod.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="O0,O1,O3r,O3")
    ap.add_argument("--torch-cpu", action="store_true")
    ap.add_argument("--n-warm", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    sel = [("P_short", 25), ("P_med", 67), ("P_long", 171)]
    prompt_ids = {n: np.tile(np.arange(2000, 2000 + L), (1, 1)).astype(np.int64) for n, L in sel}
    inputs = list(prompt_ids.items())

    out = []
    for arm in args.arms.split(","):
        try:
            if arm == "O3":
                # artifact limited to <=40 tokens on this host: use short only
                r = bench_ov(arm, inputs[:1], args.n_warm, args.n_iter, args.threads)
            else:
                r = bench_ov(arm, inputs, args.n_warm, args.n_iter, args.threads)
            out.append(r)
        except Exception as e:
            out.append({"arm": arm, "status": "FAILED", "error": str(e)[:300]})
    if args.torch_cpu:
        out.append(bench_torch_cpu(inputs, args.n_warm, args.n_iter))
    perf_path = os.path.join(ROOT, "artifacts", "metrics", "cpu_performance.json")
    if os.path.exists(perf_path):
        try:
            prev = {r.get("arm"): r for r in json.load(open(perf_path)) if r.get("arm")}
        except Exception:
            prev = {}
        for r in out:
            prev[r.get("arm")] = r
        out = list(prev.values())
    save_json(out, perf_path)
    print("saved cpu_performance.json")


if __name__ == "__main__":
    main()
