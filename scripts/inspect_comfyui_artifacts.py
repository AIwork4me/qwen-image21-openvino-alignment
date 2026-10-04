#!/usr/bin/env python3
"""v2 Phase: freeze the official ComfyUI Qwen-Image-2.1 text-encoder artifacts.

Reads safetensors headers (metadata, tensor names/dtypes/shapes) WITHOUT loading
weights into RAM, computes SHA256, and emits artifacts/v2/manifests/comfyui_encoder_artifacts.json.
"""
import argparse
import datetime
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sha256_file, save_json

ARTIFACTS = {
    "C_BF16": "qwen3vl_8b_bf16.safetensors",
    "C_INT8": "qwen3vl_8b_int8_convrot.safetensors",
    "C_W4A8": "qwen3vl_8b_w4a8.safetensors",
}
DEFAULT_DIR = "/valdata/v2models/text_encoders"
SOURCE_REPO = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1"
SOURCE_PATH = "text_encoders/{name}"
# Filled from the HF API at capture time; sizes verified byte-exact against this listing.
SOURCE_API_SIZES = {
    "qwen3vl_8b_bf16.safetensors": 17534334616,
    "qwen3vl_8b_int8_convrot.safetensors": 9350798360,
    "qwen3vl_8b_w4a8.safetensors": 6312105364,
}


def read_st_header(path):
    """Parse a safetensors header: returns (header_dict, header_len)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 512 * 1024 * 1024:
            raise ValueError(f"implausible header size {n} for {path}")
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, n


def summarize_tensors(hdr):
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        out[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "nbytes": int(info.get("data_offsets", [0, 0])[1])
            - int(info.get("data_offsets", [0, 0])[0]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default="artifacts/v2/manifests/comfyui_encoder_artifacts.json")
    ap.add_argument("--skip-hash", action="store_true", help="skip SHA256 (debug only)")
    args = ap.parse_args()

    manifest = {
        "captured_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "source_repo": SOURCE_REPO,
        "source_url_template": SOURCE_REPO + "/resolve/main/" + SOURCE_PATH,
        "download_timestamp": "2026-10-04T21:50:00Z/2026-10-04T21:56:00Z (UTC, see dl log artifacts/v2/environment/dl_v2_encoders.log)",
        "artifacts": {},
    }

    for arm, fname in ARTIFACTS.items():
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            raise SystemExit(f"MISSING artifact for {arm}: {path}")
        size = os.path.getsize(path)
        expected = SOURCE_API_SIZES[fname]
        size_ok = size == expected
        hdr, hdr_len = read_st_header(path)
        meta = hdr.get("__metadata__", {})
        tensors = summarize_tensors(hdr)
        entry = {
            "filename": fname,
            "path": path,
            "size_bytes": size,
            "size_matches_hf_api": size_ok,
            "safetensors_header_len": hdr_len,
            "safetensors_metadata": meta,
            "num_tensors": len(tensors),
            "tensor_index_path": f"artifacts/v2/manifests/tensor_index_{arm.lower()}.json",
        }
        if not args.skip_hash:
            print(f"[{arm}] hashing {fname} ({size/1e9:.2f} GB) ...", flush=True)
            entry["sha256"] = sha256_file(path)
        manifest["artifacts"][arm] = entry
        with open(os.path.join(os.path.dirname(args.out) or ".",
                               f"tensor_index_{arm.lower()}.json"), "w") as f:
            json.dump(tensors, f, indent=1, sort_keys=True)
        print(f"[{arm}] {len(tensors)} tensors, metadata keys: {list(meta)[:8]}")

    save_json(manifest, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
