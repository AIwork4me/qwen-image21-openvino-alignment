#!/usr/bin/env python3
"""Capture full environment metadata (post-ROCm-correction state) -> artifacts/environment/."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json, sha256_file

ENVDIR = os.path.join(ROOT, "artifacts", "environment")
os.makedirs(ENVDIR, exist_ok=True)


def run(cmd, out_name, shell=True):
    p = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    with open(os.path.join(ENVDIR, out_name), "w") as f:
        f.write(f"$ {cmd}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}")
    return p.returncode


def main():
    run("uname -a", "system.txt")
    run("cat /etc/os-release >> %s" % os.path.join(ENVDIR, "system.txt"), "system_append.tmp")
    os.remove(os.path.join(ENVDIR, "system_append.tmp"))
    run("lscpu", "lscpu.txt")
    run("rocminfo", "rocminfo.txt")
    run("rocm-smi", "rocm-smi.txt")
    run("rocm-smi --showmeminfo vram --json", "rocm-smi-vram.json")
    run("cat /opt/rocm/.info/version-rocm 2>/dev/null; readlink -f /opt/rocm", "rocm_version.txt")
    run("grep -E 'HIP_VERSION_(MAJOR|MINOR|PATCH)' /opt/rocm/include/hip/hip_version.h", "rocm_hip_version_header.txt")
    run("cat /proc/driver/amdgpu/version 2>/dev/null; cat /sys/module/amdgpu/version 2>/dev/null", "amdgpu_kernel_driver.txt")
    run("python -m pip freeze", "pip_freeze.txt")
    run("python -m pip list --format=json", "python_packages.json")

    summary = {}
    try:
        import torch
        summary["torch"] = torch.__version__
        summary["torch_hip"] = torch.version.hip
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["gpu_name"] = torch.cuda.get_device_name(0)
            p = torch.cuda.get_device_properties(0)
            summary["gcn_arch"] = getattr(p, "gcnArchName", None)
            summary["vram_gib"] = p.total_memory / 2 ** 30
    except Exception as e:
        summary["torch_error"] = str(e)
    try:
        import openvino
        summary["openvino"] = openvino.__version__
    except Exception as e:
        summary["openvino_error"] = str(e)
    for mod in ["transformers", "diffusers", "optimum", "nncf", "numpy", "safetensors", "PIL", "tokenizers", "accelerate", "huggingface_hub"]:
        try:
            m = __import__(mod)
            summary[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            summary[mod] = f"IMPORT_FAIL:{e}"
    run("readlink -f /opt/rocm", "rocm_symlink.txt")
    with open(os.path.join(ENVDIR, "rocm_symlink.txt")) as f:
        summary["rocm_userspace"] = f.read().strip()

    save_json(summary, os.path.join(ENVDIR, "summary.json"))
    print(json.dumps(summary, indent=2))
    print("environment capture complete ->", ENVDIR)


if __name__ == "__main__":
    main()
