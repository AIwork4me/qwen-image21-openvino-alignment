"""Shared utilities for the Qwen-Image 2.1 OpenVINO alignment validation."""
import hashlib
import json
import os

# Big data lives on the root overlay (workspace loop device is capacity-limited).
os.environ.setdefault("HF_HOME", "/valdata/hf")
os.environ.setdefault("HF_HUB_CACHE", "/valdata/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/valdata/hf/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")


def load_cfg():
    import yaml
    with open(os.path.join(ROOT, "config", "experiment.yaml")) as f:
        return yaml.safe_load(f)


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_hash(t):
    """Stable hash of a torch/numpy tensor: dtype, shape, bytes."""
    if isinstance(t, np.ndarray):
        return hashlib.sha256(f"{t.dtype}{t.shape}".encode() + t.tobytes()).hexdigest()
    return hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode() + t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def npz_save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


# ---- metrics ----
def paired_metrics(a, b, eps=1e-12):
    """Full paired-error statistics between two equal-shape float arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    d = a - b
    absd = np.abs(d)
    na = np.linalg.norm(a.reshape(-1))
    nd = np.linalg.norm(d.reshape(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = d.reshape(-1)
        denom = np.maximum(np.abs(a.reshape(-1)) + np.abs(b.reshape(-1)), eps)
        rel_elementwise = np.abs(flat) / denom
    return {
        "n": int(a.size),
        "shape": list(a.shape),
        "max_abs": float(absd.max()),
        "mean_abs": float(absd.mean()),
        "median_abs": float(np.median(absd)),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "rel_l2": float(nd / max(na, eps)),
        "norm_ratio": float(nd and np.linalg.norm(b.reshape(-1)) / max(na, eps)),
        "cosine": float(np.dot(a.reshape(-1), b.reshape(-1)) / max(np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)), eps)),
        "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
        "mean_rel_elementwise_sym": float(rel_elementwise.mean()),
        "nan_a": int(np.isnan(a).sum()), "nan_b": int(np.isnan(b).sum()),
        "inf_a": int(np.isinf(a).sum()), "inf_b": int(np.isinf(b).sum()),
    }


def per_token_cosine(a, b, eps=1e-12):
    """Per-token cosine similarity for [seq, dim] arrays; also per-channel stats."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.ndim == 2 and a.shape == b.shape
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    c = num / np.maximum(den, eps)
    return c


def stats_block(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "median": float(np.median(x)), "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max()),
    }


def per_token_summary(c):
    return {
        "min": float(np.min(c)), "p1": float(np.percentile(c, 1)), "p5": float(np.percentile(c, 5)),
        "p25": float(np.percentile(c, 25)), "median": float(np.median(c)),
        "p75": float(np.percentile(c, 75)), "p95": float(np.percentile(c, 95)),
        "p99": float(np.percentile(c, 99)), "max": float(np.max(c)),
    }


def bf16_exact_match(a, b):
    """Element match rate after both arrays are cast to bfloat16."""
    import torch
    ta = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16)
    tb = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(torch.bfloat16)
    return float((ta == tb).float().mean().item())


class ResourceMonitor:
    """Lightweight psutil-based sampler for CPU RSS/util and GPU (rocm-smi via subprocess)."""

    def __init__(self, out_csv, hz=5):
        import psutil
        self.psutil = psutil
        self.out_csv = out_csv
        self.hz = hz
        self._stop = False

    def _loop(self):
        import threading, time
        import csv
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        proc = self.psutil.Process()
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "rss_mb", "proc_cpu_pct", "sys_cpu_pct", "threads"])
            while not self._stop:
                try:
                    w.writerow([time.time(), proc.memory_info().rss / 2**20,
                                proc.cpu_percent(), self.psutil.cpu_percent(), proc.num_threads()])
                except Exception:
                    pass
                time.sleep(1.0 / self.hz)

    def start(self):
        import threading
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def stop(self):
        self._stop = True
