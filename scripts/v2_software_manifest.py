#!/usr/bin/env python3
"""v2 Phase: freeze the exact software stack (libraries + ComfyUI source)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json

COMFY_SRC = os.environ.get("V2_COMFYUI_SRC", "/valdata/comfyui-src")
COMFY_TARBALL = "/valdata/comfyui-master.tar.gz"


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def import_version(modname, attr="__version__"):
    try:
        mod = __import__(modname)
        return getattr(mod, attr, None)
    except Exception as e:
        return f"IMPORT-FAILED: {type(e).__name__}: {e}"


def main():
    import torch

    manifest = {}
    manifest["python"] = sys.version
    manifest["torch"] = {"version": torch.__version__, "hip": torch.version.hip,
                         "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    try:
        manifest["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1e6,
        }
    except Exception as e:
        manifest["gpu"] = f"unavailable: {e}"

    pkgs = ["torch", "transformers", "diffusers", "openvino", "optimum", "nncf",
            "safetensors", "numpy", "Pillow", "opencv-python-headless", "scipy",
            "comfy-kitchen", "tokenizers", "sentencepiece", "accelerate", "huggingface-hub",
            "lpips", "torchvision", "onnx", "protobuf", "pyyaml", "matplotlib", "pandas",
            "easyocr", "opencv-python", "clip", "open_clip_torch"]
    manifest["packages"] = {p: pkg_version(p) for p in pkgs}

    # ROCm userspace
    try:
        rocm_ver = subprocess.check_output(
            "cat /opt/rocm/.info/version 2>/dev/null || rocminfo --version 2>/dev/null | head -2",
            shell=True, text=True).strip()
    except Exception as e:
        rocm_ver = str(e)
    manifest["rocm"] = rocm_ver
    manifest["torch_hip_build"] = torch.version.hip

    # ComfyUI source freeze
    manifest["comfyui"] = {
        "source": "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/refs/heads/master",
        "tarball_bytes": os.path.getsize(COMFY_TARBALL) if os.path.exists(COMFY_TARBALL) else None,
        "tarball_sha256_env": "recorded in artifacts/v2/environment/ (see system_capture + sha below)",
        "extract_dir": COMFY_SRC,
        "git_repo": "not-a-git-checkout (tarball; egress proxy blocks git protocol)",
        "note": "ComfyUI master tarball captured 2026-10-04; file-level hashes in comfyui_file_hashes.json",
    }
    # hash the files that define loader semantics
    rel = [
        "comfy/ops.py", "comfy/quant_ops.py", "comfy/sd.py", "comfy/sd1_clip.py",
        "comfy/utils.py", "comfy/text_encoders/qwen3vl.py",
        "comfy/text_encoders/qwen_image21.py", "comfy/text_encoders/llama.py",
        "comfy/text_encoders/qwen35.py", "comfy/model_management.py",
    ]
    from common import sha256_file
    hashes = {}
    for r in rel:
        p = os.path.join(COMFY_SRC, r)
        hashes[r] = sha256_file(p) if os.path.exists(p) else "MISSING"
    manifest["comfyui"]["file_sha256"] = hashes
    save_json(hashes, "artifacts/v2/environment/comfyui_file_hashes.json")

    # comfy_kitchen eager reference implementation hashes (defines quant math)
    kv = os.path.dirname(__import__("comfy_kitchen").__file__)
    kfiles = ["tensor/int8.py", "tensor/int8_utils.py", "tensor/w4a8_int8.py", "tensor/base.py",
              "backends/eager/quantization.py", "backends/eager/w4a8_int8.py"]
    khashes = {f: sha256_file(os.path.join(kv, f)) for f in kfiles}
    manifest["comfy_kitchen"] = {"version": pkg_version("comfy-kitchen"),
                                 "install_path": kv, "file_sha256": khashes}
    save_json(khashes, "artifacts/v2/environment/comfy_kitchen_file_hashes.json")

    # LPIPS / OCR / SigLIP freeze (filled when those metrics are set up)
    manifest["metrics_stack"] = {
        "lpips": import_version("lpips"),
        "easyocr": import_version("easyocr"),
        "open_clip": import_version("open_clip", "__version__"),
        "transformers_clip_available": import_version("transformers") is not None,
    }

    save_json(manifest, "artifacts/v2/environment/software_manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("comfyui",)}, indent=1)[:2000])
    print("wrote artifacts/v2/environment/software_manifest.json")


if __name__ == "__main__":
    main()
