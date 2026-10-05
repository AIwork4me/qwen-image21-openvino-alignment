#!/usr/bin/env python3
"""v2 environment hard gate.

Every authoritative v2 validation entry point MUST call this check first and
abort on non-zero exit.  Verifies the mandated stack:

  ROCm SDK/runtime (pip):            7.14.1
  torch:                             2.12.0+rocm7.14.1
  torchvision:                       0.27.0+rocm7.14.1
  GPU architecture:                  gfx1100
  GPU class:                         Radeon PRO W7900 / W7900D

Exit codes: 0 = PASS, 1 = FAIL (reasons printed + written to stderr).
"""
from __future__ import annotations

import json
import subprocess
import sys

REQUIRED = {
    "rocm_runtime": "7.14.1",
    "torch": "2.12.0+rocm7.14.1",
    "torchvision": "0.27.0+rocm7.14.1",
    "gpu_arch": "gfx1100",
}


def pip_version(pkg: str) -> str | None:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def main() -> int:
    failures: list[str] = []
    report: dict = {}

    import torch  # noqa: PLC0415

    report["torch"] = torch.__version__
    report["torch_version_hip"] = str(torch.version.hip)
    if not torch.__version__.startswith(REQUIRED["torch"]):
        failures.append(f"torch {torch.__version__} != {REQUIRED['torch']}")

    import torchvision  # noqa: PLC0415
    report["torchvision"] = torchvision.__version__
    if not torchvision.__version__.startswith(REQUIRED["torchvision"]):
        failures.append(f"torchvision {torchvision.__version__} != {REQUIRED['torchvision']}")

    for pkg in ("rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-device-gfx1100"):
        v = pip_version(pkg)
        report[f"pip_{pkg}"] = v
        if v != REQUIRED["rocm_runtime"]:
            failures.append(f"pip {pkg} {v} != {REQUIRED['rocm_runtime']} (ROCm 7.14.1 mandate)")

    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is False")
        print(json.dumps(report, indent=2))
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        return 1

    report["device_count"] = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        report[f"device_{i}"] = {
            "name": torch.cuda.get_device_name(i),
            "gcn_arch": p.gcnArchName,
            "total_memory_mb": p.total_memory // (1024 * 1024),
        }
        if REQUIRED["gpu_arch"] not in (p.gcnArchName or ""):
            failures.append(f"device {i} arch {p.gcnArchName} != {REQUIRED['gpu_arch']}")
        name = torch.cuda.get_device_name(i).lower()
        if "w7900" not in name.replace(" ", ""):
            failures.append(f"device {i} {torch.cuda.get_device_name(i)} is not a Radeon PRO W7900/W7900D")

    # functional smoke: fp16 GEMM must succeed and stay finite
    a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    if not bool(torch.isfinite(c).all()):
        failures.append("GPU GEMM smoke test produced non-finite values")
    report["gpu_gemm_smoke"] = "PASS" if not failures or all("GEMM" not in f for f in failures) else "FAIL"

    print(json.dumps(report, indent=2))
    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        return 1
    print("v2 environment gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
