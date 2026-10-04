#!/usr/bin/env python3
"""Step 1: accuracy-aware re-quantization (arms O3q / O3qL).

Motivation (measured): the INT8 error chain starts at the compressed embedding table
(3.0% at layer 0) and is amplified ~2.8x in the last transformer layer. Both spots can
be excluded from INT8 at near-zero latency cost:
  O3q  = language model INT8_ASYM (same recipe as O3r) + embedding table kept FP32
  O3qL = O3q + last transformer layer (layers.35) excluded from weight compression
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_cfg, save_json

import openvino as ov
import nncf

OUT_BASE = "/valdata/models"


def export(variant):
    cfg = load_cfg()
    src = cfg["model"].get("ov_clean_export_dir") or "/valdata/models/ov_clean_fp32"
    out = os.path.join(OUT_BASE, f"ov_clean_int8_acc_{variant}")
    os.makedirs(out, exist_ok=True)
    core = ov.Core()
    lm = core.read_model(os.path.join(src, "language_model_fp32.xml"))
    ignored = None
    kwargs = dict(mode=nncf.CompressWeightsMode.INT8_ASYM, ratio=1.0, group_size=-1)
    if variant == "full":  # O3qL: exclude the last transformer block
        ignored = nncf.IgnoredScope(patterns=[r"__module\.lm\.layers\.35\..*"])
    if variant == "mixed":  # O3m: sensitivity-aware mixed precision (data-free metric)
        from nncf.parameters import SensitivityMetric
        kwargs.update(ratio=0.6, all_layers=True,
                      sensitivity_metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR)
    if ignored is not None:
        kwargs["ignored_scope"] = ignored
    t0 = time.time()
    compressed = nncf.compress_weights(lm, **kwargs)
    ov.save_model(compressed, os.path.join(out, "language_model_int8.xml"), compress_to_fp16=False)
    sz = os.path.getsize(os.path.join(out, "language_model_int8.bin")) / 2 ** 30
    save_json({"source": src, "tool": f"nncf {nncf.__version__} compress_weights",
               "mode": "INT8_ASYM", "ratio": 1.0, "group_size": -1,
               "ignored_scope_patterns": (["__module\\.lm\\.layers\\.35\\..*"] if ignored else []),
               "mixed_precision": {"ratio": 0.6, "metric": "weight_quantization_error"} if variant == "mixed" else None,
               "embedding": "FP32 table (no int8) — numpy gather",
               "variant": variant, "bin_gib": sz},
              os.path.join(out, "conversion_manifest.json"))
    print(f"[{variant}] compressed in {time.time()-t0:.0f}s -> {out} ({sz:.2f} GiB)")


def verify_exclusion(variant):
    """Prove the exclusion actually happened: count int8 (u8/s8) weight Constants in the
    saved IR and attribute each to the nearest transformer-layer scope via its consumers.
    (nncf inserts Convert/dequant ops between the Constant and the MatMul, so we cannot
    look at MatMul inputs directly.)"""
    import re
    import numpy as np
    out = os.path.join(OUT_BASE, f"ov_clean_int8_acc_{variant}")
    core = ov.Core()
    m = core.read_model(os.path.join(out, "language_model_int8.xml"))
    per_layer_int8 = {}
    total_int8 = 0
    for op in m.get_ordered_ops():
        if op.get_type_name() != "Constant":
            continue
        et = op.get_output_element_type(0).get_type_name()
        if et not in ("u8", "i8"):
            continue
        total_int8 += 1
        # walk consumers up to 3 hops to find a layer-scoped op name
        scope = "other"
        frontier = [op]
        for _ in range(3):
            nxt = []
            for n in frontier:
                for out_p in n.outputs():
                    for inp in out_p.get_target_inputs():
                        c = inp.get_node()
                        fn = c.get_friendly_name()
                        mm = re.search(r"layers\.(\d+)\.", fn)
                        if mm:
                            scope = f"layers.{mm.group(1)}"
                        else:
                            nxt.append(c)
            frontier = nxt
            if scope != "other":
                break
        per_layer_int8[scope] = per_layer_int8.get(scope, 0) + 1
    n35 = per_layer_int8.get("layers.35", 0)
    rest = total_int8 - n35 - per_layer_int8.get("other", 0)
    print(f"[{variant}] int8 weight Constants total={total_int8}; in layers.35={n35}; "
          f"in layers.0-34={rest}; other-scope={per_layer_int8.get('other', 0)}")
    return {"total_int8": total_int8, "int8_in_layer35": n35, "int8_in_layers_0_34": rest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["emb", "full", "mixed"], required=True)
    args = ap.parse_args()
    export(args.variant)
    stats = verify_exclusion(args.variant)
    print("DONE", args.variant, stats)


if __name__ == "__main__":
    main()
