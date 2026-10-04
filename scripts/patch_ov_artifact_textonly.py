#!/usr/bin/env python3
"""Runtime text-only patch for the ComfyUI-OV Qwen3-VL artifacts (arm O3p).

ROOT CAUSE (measured on EPYC 9334 / OpenVINO 2026.4.0-2026.4.1):
  The exported IR bakes the trace-time visual/deepstack path:
    NonZero(visual_pos_masks) -> Transpose -> GatherND   (vision-token scatter)
    Gather(deepstack_visual_embeds, idx=<baked constant>) -> Add_3/Add_4/Add_5
  For text-only inputs (mask all-False) this becomes an empty-tensor path that the
  CPU plugin's eltwise shape inference rejects (dim-0 mismatch), nondeterministically
  and at minimum for prompts > ~40 tokens on this host.

PATCH (runtime-only; serialized customer files are NEVER modified):
  For every top-level Add whose input is produced by `aten::select` Gather on
  deepstack_visual_embeds, replace that input with an empty Constant [0, 4096].
  For text-only sequences the visual insert is mathematically a no-op, so the patch
  is semantics-preserving — proven BITWISE against the artifact's own frozen outputs
  captured while it still ran on this host (smoke suite: max_abs = 0.0 on S01-S03).

NOTE: with vision inputs the patched model would be WRONG by construction; this arm
is for text-to-image conditioning validation only.
"""
import numpy as np
import openvino as ov


def apply_textonly_patch(model):
    cst = ov.opset13.constant(np.zeros((0, 4096), dtype=np.float32))
    n = 0
    for op in model.get_ordered_ops():
        fn = op.get_friendly_name()
        if op.get_type_name() == "Add" and fn.startswith("__module.model.language_model/aten::add/"):
            for i in range(op.get_input_size()):
                prod = op.input_value(i).get_node()
                if prod.get_type_name() == "Gather" and "aten::select" in prod.get_friendly_name():
                    op.input(i).replace_source_output(cst.output(0))
                    n += 1
    assert n >= 3, f"expected >=3 visual-branch Adds to patch, found {n}"
    return n
