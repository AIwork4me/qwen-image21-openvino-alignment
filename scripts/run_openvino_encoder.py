#!/usr/bin/env python3
"""Run OpenVINO CPU Qwen3-VL-8B text encoder arms (O0 clean-export / O1 runtime-bf16 /
O3 customer INT8 artifact) with layer-wise instrumentation via add_outputs.

Feeds EXACTLY the frozen inputs from artifacts/inputs (same input_ids as ROCm arms).
Saves npz per prompt in the same schema as run_rocm_encoder.py:
  hidden_states [37, seq, 4096] fp32  (entry 0 = token embeddings)
  prenorm, postnorm, prompt_embeds, prompt_embeds_bf16_view, drop_idx
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, npz_save, save_json, tensor_hash

import openvino as ov
import openvino.properties as props

CUSTOMER_INT8 = "/workspace/qwen3vl-cpu-benchmark/models/qwen3vl-openvino-int8"
CUSTOMER_FP16 = "/root/models/qwen3vl-openvino-fp16"
CLEAN_FP32 = "/valdata/models/ov_clean_fp32"


def find_prenorm_anchor(model):
    """Return op names (pre-norm producer, final-norm first op) using name/edge analysis."""
    ordered = model.get_ordered_ops()
    by_name = {}
    for op in ordered:
        by_name[op.get_friendly_name()] = op
    # pre-norm producer pattern from prior forensic work: layers.35/aten::add/Add_1
    cand = [n for n in by_name if re.fullmatch(r"__module\.model\.language_model\.layers\.35/aten::add/Add_1", n)]
    norm_first = [n for n in by_name if n.startswith("__module.model.language_model.norm/")]
    assert cand, "pre-norm anchor not found"
    return cand[0], (norm_first[0] if norm_first else None)


def layer_output_ops(model, n_layers=36):
    """Map transformer layer index -> op producing its final residual output."""
    import re as _re
    ops = {}
    for op in model.get_ordered_ops():
        n = op.get_friendly_name()
        m = _re.fullmatch(r"__module\.model\.language_model\.layers\.(\d+)/aten::add/Add_1", n)
        if m:
            idx = int(m.group(1))
            if idx not in ops:
                ops[idx] = op
    missing = [i for i in range(n_layers) if i not in ops]
    assert not missing, f"missing layer ops: {missing[:5]}..."
    return [ops[i] for i in range(n_layers)]


def build_feed(model_inputs, input_ids, embeds):
    n = input_ids.shape[1]
    feed = {}
    for i, inp in enumerate(model_inputs):
        name = list(inp.names)[0] if inp.names else f"input_{i}"
        et = inp.get_element_type().get_type_name()
        shape = inp.get_partial_shape()
        ranks = len(shape)
        if name in ("inputs_embeds",) or (et == "f32" and ranks == 3):
            feed[i] = embeds.astype(np.float32)
        elif name == "attention_mask" or (et == "i64" and ranks == 2):
            feed[i] = np.ones_like(input_ids)
        elif name == "position_ids" or (et == "i64" and ranks == 3):
            ar = np.arange(n, dtype=np.int64)
            feed[i] = np.stack([ar, ar, ar]).reshape(3, 1, n)
        elif name == "visual_pos_masks":
            feed[i] = np.zeros((1, n), dtype=bool)
        elif name == "deepstack_visual_embeds":
            feed[i] = np.zeros((1, 0, 4096), dtype=np.float32)
        elif name == "beam_idx":
            feed[i] = np.array([0], dtype=np.int32)
        else:
            raise RuntimeError(f"unhandled model input: {name} ({et}, rank {ranks})")
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="O0 | O1 | O3 | O1F")
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--artifact", default=None, help="override artifact dir")
    ap.add_argument("--runtime-precision", default=None, help="f32 | bf16 (inference_precision hint)")
    ap.add_argument("--hint", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg()
    if args.artifact:
        art = args.artifact
    elif args.arm == "O3":
        art = CUSTOMER_INT8
    elif args.arm == "O1F":
        art = CUSTOMER_FP16
    elif args.arm == "O3r":
        art = "/valdata/models/ov_clean_int8"
    elif args.arm == "O3p":
        art = CUSTOMER_INT8  # real customer artifact + runtime text-only patch
    elif args.arm == "O3q":
        art = "/valdata/models/ov_clean_int8_acc_emb"
    elif args.arm == "O3qL":
        art = "/valdata/models/ov_clean_int8_acc_full"
    else:
        art = cfg["model"].get("ov_clean_export_dir") or CLEAN_FP32

    hint = args.hint or ("ACCURACY" if args.arm in ("O0", "O1", "O1F") else "ACCURACY")
    runtime_prec = args.runtime_precision or ("bf16" if args.arm == "O1" else "f32")

    core = ov.Core()
    t0 = time.time()
    if args.arm in ("O3", "O1F", "O3p"):
        # ComfyUI-OV style artifact: patch with add_outputs (runtime-only; files untouched)
        lm = core.read_model(os.path.join(art, "openvino_language_model.xml"))
        if args.arm == "O3p":
            from patch_ov_artifact_textonly import apply_textonly_patch
            n_patch = apply_textonly_patch(lm)
            print(f"[O3p] applied runtime text-only patch to {n_patch} visual-branch Adds")
        pre_name, norm_name = find_prenorm_anchor(lm)
        lops = layer_output_ops(lm)
        pre_op = next(op for op in lm.get_ordered_ops() if op.get_friendly_name() == pre_name)
        norm_op = next(op for op in lm.get_ordered_ops() if op.get_friendly_name() == norm_name) if norm_name else None
        extra = [op.output(0) for op in lops] + [pre_op.output(0)]
        if norm_op is not None:
            # post-norm: attach to the norm's output — norm has multiple ops; use its final mul
            norm_outs = [op for op in lm.get_ordered_ops()
                         if op.get_friendly_name().startswith("__module.model.language_model.norm/")
                         and op.get_type_name() in ("Multiply", "Divide", "MVN", "Subtract")]
            extra.append(norm_outs[-1].output(0))
        lm.add_outputs(extra)
        emb = core.read_model(os.path.join(art, "openvino_text_embeddings_model.xml"))
        is_wrapper = False
    elif args.arm in ("O3r", "O3q", "O3qL"):
        lm = core.read_model(os.path.join(art, "language_model_int8.xml"))
        emb = None
        is_wrapper = True
    else:
        lm = core.read_model(os.path.join(art, "language_model_fp32.xml"))
        emb = None
        is_wrapper = True
    read_s = time.time() - t0

    p = {"EXECUTION_MODE_HINT": "ACCURACY"}
    if hint in ("LATENCY", "THROUGHPUT"):
        p["PERFORMANCE_HINT"] = hint
    if runtime_prec == "bf16":
        p["INFERENCE_PRECISION_HINT"] = "bf16"
    elif runtime_prec == "f32":
        p["INFERENCE_PRECISION_HINT"] = "f32"
    if args.threads:
        p["INFERENCE_NUM_THREADS"] = args.threads

    t0 = time.time()
    compiled = core.compile_model(lm, "CPU", p)
    compile_s = time.time() - t0
    emb_compiled = core.compile_model(emb, "CPU", p) if emb is not None else None

    exec_devs = "CPU"
    try:
        ip = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        ip = "n/a"
    try:
        hint_got = str(compiled.get_property("PERFORMANCE_HINT"))
    except Exception:
        hint_got = "n/a"
    try:
        emode_got = str(compiled.get_property("EXECUTION_MODE_HINT"))
    except Exception:
        emode_got = "n/a"

    out_names = [list(o.names)[0] if o.names else f"out_{i}" for i, o in enumerate(compiled.outputs)]
    print(f"[{args.arm}] openvino={ov.__version__} read={read_s:.1f}s compile={compile_s:.1f}s "
          f"perf_hint={hint_got} exec_mode={emode_got} inference_precision={ip} outputs={len(out_names)}")
    print(f"[{args.arm}] output names (first/last): {out_names[:3]} ... {out_names[-3:]}")

    prompts = json.load(open(os.path.join(ROOT, f"prompts/{args.suite}.json")))["prompts"]
    outdir = os.path.join(ROOT, "artifacts", "tensors", args.arm)
    os.makedirs(outdir, exist_ok=True)

    entries = []
    snap_path = cfg["model"]["snapshot_path"]
    for pr in prompts:
        pid = pr["id"]
        z = np.load(os.path.join(ROOT, "artifacts", "inputs", f"{pid}.npz"))
        input_ids = z["input_ids"].astype(np.int64)
        drop_idx = int(z["drop_idx"])
        t0 = time.time()
        if is_wrapper:
            if args.arm == "O3r":
                if "_int8_tbl" not in dir():
                    tz = np.load("/valdata/models/ov_clean_int8/embed_table_int8.npz")
                    int8_tbl = (tz["q"], tz["zero_point"], tz["scale"])
                embeds = embed_lookup_fp32(snap_path, input_ids, int8_table=int8_tbl)
            else:
                # O0 / O1 / O3q / O3qL: exact fp32 embedding-table gather
                embeds = embed_lookup_fp32(snap_path, input_ids)
        else:
            embeds = np.asarray(emb_compiled({0: input_ids})[0], dtype=np.float32)
        emb_s = time.time() - t0
        feed = build_feed(compiled.inputs, input_ids, embeds)
        req = compiled.create_infer_request()
        req.reset_state()
        t0 = time.time()
        res = req.infer(feed)
        infer_s = time.time() - t0

        outs = [np.asarray(res[i], dtype=np.float32) for i in range(len(compiled.outputs))]
        if is_wrapper:
            # wrapper export: outputs = 37 hidden states (0=embeddings, 1..36=layers) + prenorm + postnorm
            hs_list = outs[:37]
            prenorm, postnorm = outs[37], outs[38]
        else:
            # artifact + add_outputs: [logits, layer0..layer35, postnorm]
            # NOTE: the pre-norm anchor IS layer-35's output op -> add_outputs dedupes it.
            hs_list = outs[1:37]
            prenorm = outs[36]          # layer-35 residual output == pre-final-norm hidden state
            postnorm = outs[37]
            hs_list = [embeds] + hs_list
        want = (1, input_ids.shape[1], 4096)
        for i, h in enumerate(hs_list):
            assert h.shape == want, f"entry {i} shape {h.shape} != {want}"
        hs = np.stack([h[0] for h in hs_list])  # [37, seq, D]
        pre = prenorm[0] if prenorm.ndim == 3 else prenorm
        post = postnorm[0] if postnorm.ndim == 3 else postnorm
        assert hs.shape[0] == 37, hs.shape
        assert not np.allclose(pre, post), "postnorm equals prenorm — norm output pick is wrong"

        prompt_embeds = pre[drop_idx:]
        pe_bf16_view = torch_from_bf16(prompt_embeds)

        npz_save(os.path.join(outdir, f"{pid}.npz"),
                 hidden_states=hs, prenorm=pre, postnorm=post,
                 prompt_embeds=prompt_embeds, prompt_embeds_bf16_view=pe_bf16_view,
                 drop_idx=np.array(drop_idx), token_embeddings=embeds[0])
        entries.append({"pid": pid, "seq_len": int(hs.shape[1]), "kept_len": int(prompt_embeds.shape[0]),
                        "embed_s": emb_s, "infer_s": infer_s,
                        "prenorm_hash": tensor_hash(pre), "pe_hash": tensor_hash(prompt_embeds)})
        print(f"[{args.arm}][{pid}] seq={hs.shape[1]} kept={prompt_embeds.shape[0]} "
              f"emb={emb_s:.3f}s infer={infer_s:.3f}s")

    save_json({"arm": args.arm, "artifact": art, "impl": "openvino_cpu", "exec_mode": emode_got,
               "openvino": ov.__version__, "hint": hint_got, "inference_precision": ip,
               "runtime_precision_requested": runtime_prec, "compile_s": compile_s, "read_s": read_s,
               "model_files": {"language_model": sha_art(art), "entries": entries}},
              os.path.join(outdir, f"meta_{args.arm}_{args.suite}.json"))
    print("done", args.arm)


def embed_lookup_fp32(snapshot, input_ids, int8_table=None):
    """Token-embedding gather. int8_table: (q, zero_point, scale) row-wise int8-asym
    dequant path reproducing the customer artifact's compressed embedding table."""
    if int8_table is not None:
        q, zp, scale = int8_table
        rows = input_ids.reshape(-1)
        e = q[rows].astype(np.float32) * scale[rows] + zp[rows]
        return e.reshape(input_ids.shape[0], input_ids.shape[1], -1).astype(np.float32)
    import glob
    from safetensors.torch import safe_open as safe_open_torch
    key = "model.language_model.embed_tokens.weight"
    table = None
    for f in sorted(glob.glob(os.path.join(snapshot, "text_encoder", "model-*.safetensors"))):
        with safe_open_torch(f, framework="pt") as sf:
            if key in sf.keys():
                table = sf.get_tensor(key).float().numpy()  # bf16 storage -> fp32 (same upcast as R1)
                break
    assert table is not None, "embed_tokens not found"
    return table[input_ids.reshape(-1)].reshape(input_ids.shape[0], input_ids.shape[1], -1).astype(np.float32)



    import torch
    return torch.from_numpy(a).to(torch.bfloat16).float().numpy()


def torch_from_bf16(a):
    import torch
    return torch.from_numpy(a).to(torch.bfloat16).float().numpy()


def sha_art(art):
    from common import sha256_file
    for f in ("openvino_language_model.xml", "language_model_int8.xml", "language_model_fp32.xml"):
        p = os.path.join(art, f)
        if os.path.exists(p):
            return {"file": f, "sha256": sha256_file(p)}
    return None


if __name__ == "__main__":
    main()
