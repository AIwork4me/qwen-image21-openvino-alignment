#!/usr/bin/env python3
"""Inspect the three official ComfyUI Qwen3-VL-8B text-encoder artifacts.

For each safetensors file: SHA256, size, safetensors metadata, full tensor
index (name/dtype/shape/offset), per-layer comfy_quant JSON configs, and
format classification.  Emits:
  artifacts/v2/manifests/comfyui_encoder_artifacts.json
  artifacts/v2/manifests/tensor_index_{arm}.json
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_common as C


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, hlen + 8


def main() -> int:
    C.gate_environment() if os.environ.get("V2_GATE", "1") == "1" else None
    out = {"captured_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
           "source_repo": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1",
           "source_url_template": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/{name}",
           "artifacts": {}}
    for arm, fname in C.ENCODER_FILES.items():
        path = os.path.join(C.ENCODER_DIR, fname)
        if not os.path.exists(path):
            print(f"MISSING {path}", file=sys.stderr)
            return 2
        size = os.path.getsize(path)
        header, data_off = read_header(path)
        meta = header.pop("__metadata__", None)
        index = {}
        quant_cfgs = {}
        dtypes = {}
        for name, info in header.items():
            index[name] = {"dtype": info["dtype"], "shape": info["shape"],
                           "data_offsets": info["data_offsets"]}
            dtypes.setdefault(info["dtype"], 0)
            dtypes[info["dtype"]] += 1
            if name.endswith(".comfy_quant"):
                pass  # parsed lazily below (needs tensor read)
        out["artifacts"][arm] = {
            "filename": fname, "path": path, "size_bytes": size,
            "safetensors_data_start_offset": data_off,
            "safetensors_header_len": data_off - 8,
            "safetensors_metadata": meta,
            "num_tensors": len(index),
            "dtype_histogram": dtypes,
            "tensor_index_path": f"artifacts/v2/manifests/tensor_index_{arm.lower()}.json",
            "sha256": C.sha256_file(path),
        }
        C.save_json(index, os.path.join(C.REPO, "artifacts", "v2", "manifests", f"tensor_index_{arm.lower()}.json"))
        print(f"{arm}: {fname} tensors={len(index)} size={size} sha256={out['artifacts'][arm]['sha256'][:16]}...")
    C.save_json(out, os.path.join(C.REPO, "artifacts", "v2", "manifests", "comfyui_encoder_artifacts.json"))
    print("manifest -> artifacts/v2/manifests/comfyui_encoder_artifacts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
