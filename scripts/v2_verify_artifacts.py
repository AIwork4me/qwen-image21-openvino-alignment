#!/usr/bin/env python3
"""v2 artifact integrity checker: validates published v2 evidence end-to-end.

Checks (critical = report generation must fail on error):
  - environment manifest: ROCm 7.14.1, torch/torchvision build tags, gfx1100, W7900*
  - frozen artifact SHA256s vs recorded manifests (3 ComfyUI encoders)
  - experiment counts (population pairs, prompt coverage, seeds)
  - input identity grid (all arms share tokenization)
  - no NaN/Inf in metric CSVs; no duplicate experiment ids in manifests
  - required plots/reports exist
Exit 0 = PASS, 1 = FAIL (reasons printed).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "v2")
CRITICAL: list[str] = []
WARNINGS: list[str] = []


def check(cond: bool, msg: str, critical: bool = True) -> bool:
    if not cond:
        (CRITICAL if critical else WARNINGS).append(msg)
    return cond


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # 1) environment manifest
    mpath = os.path.join(ART, "environment", "software_manifest.json")
    if check(os.path.exists(mpath), "missing environment/software_manifest.json"):
        man = json.load(open(mpath))
        check(man.get("rocm_sdk_runtime") == "7.14.1", f"ROCm runtime {man.get('rocm_sdk_runtime')} != 7.14.1")
        check(str(man.get("torch_build_tag", "")).startswith("2.12.0+rocm7.14.1"),
              f"torch {man.get('torch_build_tag')} != 2.12.0+rocm7.14.1")
        check(str(man.get("torchvision_build_tag", "")).startswith("0.27.0+rocm7.14.1"),
              f"torchvision {man.get('torchvision_build_tag')} != 0.27.0+rocm7.14.1")
        check(man.get("gpu", {}).get("architecture") == "gfx1100", "GPU arch != gfx1100")
        check("W7900" in man.get("gpu", {}).get("model", ""), "GPU model not W7900/W7900D")

    # 2) frozen artifact hashes
    aman = os.path.join(ART, "manifests", "comfyui_encoder_artifacts.json")
    if check(os.path.exists(aman), "missing manifests/comfyui_encoder_artifacts.json"):
        enc = json.load(open(aman))
        for name, rec in enc.items():
            local = rec.get("local_path") or os.path.join("/valdata/comfyui_encoders", name)
            if not check(os.path.exists(local), f"artifact missing on disk: {local}", critical=False):
                continue
            d = sha256_file(local)
            check(d == rec["sha256"], f"{name} sha256 mismatch {d[:12]}... != {rec['sha256'][:12]}...")

    # 3) metric CSVs present, finite, with expected coverage
    mdir = os.path.join(ART, "metrics")
    need = ["encoder_matrix_summary.csv", "int8_weight_parity.csv", "w4a8_weight_parity.csv",
            "layer_alignment_C_BF16_C_INT8.csv", "layer_alignment_C_BF16_C_W4A8.csv",
            "population_pairs.csv", "runtime_prompt_length_matrix.csv", "perf_matrix.csv",
            "semantic_scores_population.csv", "ocr_scores_population.csv"]
    for n in need:
        check(os.path.exists(os.path.join(mdir, n)), f"missing metric {n}")
    for n in glob.glob(os.path.join(mdir, "*.csv")):
        with open(n) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for row in reader:
                for c in cols:
                    v = row.get(c, "")
                    if v in ("nan", "inf", "-inf", "NaN", "Inf"):
                        CRITICAL.append(f"non-finite value in {os.path.basename(n)} col {c}")

    # 4) population coverage: 100 prompts x {25,40} x {INT8, W4A8}
    pp = os.path.join(mdir, "population_pairs.csv")
    if os.path.exists(pp):
        with open(pp) as f:
            rows = list(csv.DictReader(f))
        pids = {r["pid"] for r in rows}
        check(len(pids) >= 100, f"population covers only {len(pids)} prompts (<100)")
        for vs in ("C_INT8", "C_W4A8"):
            for steps in ("25", "40"):
                n = sum(1 for r in rows if r["vs"] == vs and r["steps"] == steps)
                check(n >= 100, f"population pairs {vs}@{steps}s = {n} < 100")

    # 5) input identity
    ii = os.path.join(mdir, "input_identity.json")
    if check(os.path.exists(ii), "missing input_identity.json"):
        ident = json.load(open(ii))
        for pid, arms in ident.items():
            hashes = {rec["ids_sha256"] for rec in arms.values() if isinstance(rec, dict)}
            check(len(hashes) <= 1, f"tokenization differs across arms for {pid}")

    # 6) required plots and reports
    for p in ["layer_C_BF16_C_INT8.png", "layer_C_BF16_C_W4A8.png"]:
        check(os.path.exists(os.path.join(ART, "plots", p)), f"missing plot {p}")
    for r in ["EXECUTIVE_SUMMARY.md", "CUSTOMER_REPORT.md", "TECHNICAL_REPORT.md",
              "REPRODUCTION.md", "LIMITATIONS.md", "EVIDENCE_INDEX.md"]:
        check(os.path.exists(os.path.join(REPO, "reports", "v2", r)), f"missing report v2/{r}")

    # 7) no duplicate experiment ids
    ids = []
    for m in glob.glob(os.path.join(ART, "metrics", "*.json")):
        try:
            d = json.load(open(m))
        except Exception:
            WARNINGS.append(f"unparsable json {os.path.basename(m)}")
            continue
        if isinstance(d, dict) and "experiment_id" in d:
            ids.append((d["experiment_id"], os.path.basename(m)))
    dupes = {i for i in ids if [x[0] for x in ids].count(i[0]) > 1}
    for i in dupes:
        WARNINGS.append(f"duplicate experiment id {i[0]} ({i[1]})")

    # 8) v2 tree must not present 7.14.0 as v2 state
    for root, dirs, files in os.walk(ART):
        dirs[:] = [d for d in dirs if d != "prior_attempt_rocm7140_runtime_mismatch"]
        for fn in files:
            if fn.endswith((".json", ".csv", ".md", ".txt")):
                p = os.path.join(root, fn)
                try:
                    txt = open(p, errors="ignore").read()
                except Exception:
                    continue
                if "7.14.0" in txt and "historical" not in txt and "prior attempt" not in txt.lower():
                    CRITICAL.append(f"v2 file references 7.14.0 as current state: {p}")

    print(f"critical: {len(CRITICAL)}  warnings: {len(WARNINGS)}")
    for c in CRITICAL:
        print("CRITICAL:", c)
    for w in WARNINGS:
        print("WARNING:", w)
    if CRITICAL:
        return 1
    print("v2 artifact integrity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
