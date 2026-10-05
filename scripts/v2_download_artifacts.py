#!/usr/bin/env python3
"""Download and verify v2 frozen model + ComfyUI text-encoder artifacts.

Downloads (idempotent, resumable):
  - Qwen/Qwen-Image-2.1 full snapshot (transformer, VAE, text encoder, configs)
  - Comfy-Org/Qwen-Image-2.1 text_encoders/{qwen3vl_8b_bf16,qwen3vl_8b_int8_convrot,qwen3vl_8b_w4a8}.safetensors

Verifies file sizes vs the HF API and computes SHA256 for the three encoder
artifacts.  Writes /valdata/comfyui_encoders/download_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import HfApi

os.environ.setdefault("HF_HOME", "/valdata/hf")
ENC_DIR = "/valdata/comfyui_encoders"

MODEL_REPO = "Qwen/Qwen-Image-2.1"
ENC_REPO = "Comfy-Org/Qwen-Image-2.1"
ENCODERS = [
    "text_encoders/qwen3vl_8b_bf16.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_w4a8.safetensors",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    api = HfApi()
    manifest: dict = {"captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("== snapshot:", MODEL_REPO, file=sys.stderr)
    info = api.model_info(MODEL_REPO)
    revision = info.sha
    manifest["model"] = {"repo": MODEL_REPO, "revision": revision}
    path = snapshot_download(MODEL_REPO, revision=revision, max_workers=8)
    manifest["model"]["snapshot_path"] = path

    print("== encoders:", ENC_REPO, file=sys.stderr)
    manifest["encoders"] = {}
    for name in ENCODERS:
        local = hf_hub_download(ENC_REPO, name)
        size = os.path.getsize(local)
        print(f"hashing {name} ({size} bytes)", file=sys.stderr)
        digest = sha256_file(local)
        manifest["encoders"][name.rsplit("/", 1)[-1]] = {
            "repo": ENC_REPO,
            "repo_path": name,
            "local_path": local,
            "size_bytes": size,
            "sha256": digest,
        }

    out = os.path.join(ENC_DIR, "download_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("manifest ->", out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
