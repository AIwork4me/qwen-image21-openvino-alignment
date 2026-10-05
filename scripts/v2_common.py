"""Shared utilities for v2 validation (paths, arms, metrics, manifests)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALDATA = os.environ.get("V2_VALDATA", "/valdata")
HF_HOME = os.path.join(VALDATA, "hf")
ENCODER_DIR = os.path.join(VALDATA, "comfyui_encoders")
MODEL_SNAPSHOT = os.path.join(HF_HOME, "hub", "models--Qwen--Qwen-Image-2.1", "snapshots")
COMFYUI_ROOT = "/workspace/ComfyUI"
ART = os.path.join(REPO, "artifacts", "v2")

ARMS = {
    "R_FP32": {"family": "reference", "runtime": "pytorch/rocm", "dtype": "fp32"},
    "R_BF16": {"family": "reference", "runtime": "pytorch/rocm", "dtype": "bf16"},
    "C_BF16": {"family": "comfyui", "runtime": "comfyui/rocm", "artifact": "qwen3vl_8b_bf16.safetensors"},
    "C_INT8": {"family": "comfyui", "runtime": "comfyui/rocm", "artifact": "qwen3vl_8b_int8_convrot.safetensors"},
    "C_W4A8": {"family": "comfyui", "runtime": "comfyui/rocm", "artifact": "qwen3vl_8b_w4a8.safetensors"},
    "O_FP32": {"family": "openvino", "runtime": "openvino/cpu", "precision": "FP32"},
    "O_BF16": {"family": "openvino", "runtime": "openvino/cpu", "precision": "BF16"},
    "O_INT8_EXACT": {"family": "openvino", "runtime": "openvino/cpu", "precision": "INT8", "parity_goal": "exact"},
    "O_INT8_ALT": {"family": "openvino", "runtime": "openvino/cpu", "precision": "INT8", "parity_goal": "alternative"},
    "O_W4A8_EXACT": {"family": "openvino", "runtime": "openvino/cpu", "precision": "W4A8-like", "parity_goal": "exact"},
    "O_W4_ALT": {"family": "openvino", "runtime": "openvino/cpu", "precision": "W4-like", "parity_goal": "alternative"},
}

ENCODER_FILES = {
    "C_BF16": "qwen3vl_8b_bf16.safetensors",
    "C_INT8": "qwen3vl_8b_int8_convrot.safetensors",
    "C_W4A8": "qwen3vl_8b_w4a8.safetensors",
}


def setup_hf_env():
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def model_snapshot_path() -> str:
    snaps = os.listdir(MODEL_SNAPSHOT)
    assert len(snaps) >= 1, "model snapshot missing"
    return os.path.join(MODEL_SNAPSHOT, snaps[0])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    d = (a - b).norm() / a.norm().clamp(min=1e-12)
    return float(d)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64, b64 = a.double(), b.double()
    a32, b32 = a.float(), b.float()
    d = a64 - b64
    ad = d.abs()
    out = {
        "shape_a": list(a.shape), "shape_b": list(b.shape),
        "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "mean_abs_err": float(ad.mean()),
        "median_abs_err": float(ad.median()),
        "max_abs_err": float(ad.max()) if ad.numel() else 0.0,
        "rmse": float(d.pow(2).mean().sqrt()),
        "rel_l2": rel_l2(a, b),
        "cosine": cosine(a, b),
        "norm_ratio": float(b64.norm() / a64.norm().clamp(min=1e-12)),
        "sign_agreement": float((torch.sign(a64) == torch.sign(b64)).float().mean()),
        "nan": bool(torch.isnan(b32).any()), "inf": bool(torch.isinf(b32).any()),
    }
    if a.dim() >= 2:
        pt_c = torch.nn.functional.cosine_similarity(a64, b64, dim=-1)
        pt_l = (a64 - b64).norm(dim=-1) / a64.norm(dim=-1).clamp(min=1e-12)
        out["per_token_cosine_min"] = float(pt_c.min())
        out["per_token_cosine_mean"] = float(pt_c.mean())
        out["per_token_rel_l2_mean"] = float(pt_l.mean())
        out["per_token_rel_l2_max"] = float(pt_l.max())
    return out


def stats_block(x: list[float]) -> dict:
    if not x:
        return {"n": 0}
    arr = np.asarray(x, dtype=np.float64)
    return {
        "n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()), "max": float(arr.max()),
    }


def experiment_manifest(experiment_id: str, extra: dict | None = None) -> dict:
    import subprocess
    m = {
        "experiment_id": experiment_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gfx_arch": torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else None,
        "torch": torch.__version__, "torch_hip": str(torch.version.hip),
    }
    if extra:
        m.update(extra)
    return m


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def load_prompts(suite: str) -> list[dict]:
    with open(os.path.join(REPO, "prompts", f"{suite}.json")) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("prompts", data)


def gate_environment() -> None:
    """Hard gate: refuse to run authoritative experiments on a wrong stack."""
    import torch as _t
    assert _t.__version__.startswith("2.12.0+rocm7.14.1"), _t.__version__
    assert _t.cuda.is_available()
    p = _t.cuda.get_device_properties(0)
    assert "gfx1100" in (p.gcnArchName or ""), p.gcnArchName
    name = _t.cuda.get_device_name(0).lower()
    assert "w7900" in name or "w 7900" in name, name
