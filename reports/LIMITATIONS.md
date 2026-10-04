# Limitations and Deviations

This validation is honest about its boundaries. The following limitations apply when
interpreting the results.

## Environment

1. **GPU is a Radeon PRO W7900D** (gfx1100), the dual-slot W7900 variant. All conclusions
   are for gfx1100 + ROCm 7.14.0 userspace; the "D" suffix does not change the compute
   architecture.
2. **ROCm userspace correction**: the machine originally exposed ROCm 7.2.1 userspace with
   a torch `2.13.0+rocm7.14.0` build. We installed the provided therock ROCm 7.14.0 dist
   (matching torch's HIP 7.14.60850) and re-pointed `/opt/rocm` (reversible symlink change,
   recorded in `artifacts/environment/rocm_symlink*.txt`). All measurements ran after the
   correction.
3. Single machine, single GPU, single CPU. No multi-run variance across hosts.

## Artifacts

4. **Customer INT8 artifact (O3) is host-limited**: on this host (EPYC 9334 / Zen4,
   OpenVINO 2026.4.0 and 2026.4.1), `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov`'s IR fails with
   an eltwise shape-inference error (`language_model/aten::add/Add_3`) for prompts whose
   token sequences exceed ~40 tokens, and for some synthetic token id patterns even below
   that. The same artifact ran 185-token prompts on the prior Zen5 hosts. We therefore:
   - measured O3 on the smoke suite (≤40 tokens) where it works;
   - reproduced the quantization recipe locally (**O3r**: NNCF INT8_ASYM weights, ratio 1.0,
     group_size −1, plus row-wise int8-asym embedding table) on the clean export;
   - verified O3r ≈ O3 numerically (O0↔O3: rel L2 7.0% / O0↔O3r: 7.0%; R0↔O3: 9.7% /
     R0↔O3r: 9.8% on identical prompts) and used O3r for full-coverage suites.
   This artifact IR issue is itself a production-relevant finding for the customer.
5. **O2** (a distinct customer "optimized, non-quantized" OpenVINO artifact) was searched
   for and **not found** on this machine; labeled unavailable. The FP16-storage artifact at
   `/root/models/qwen3vl-openvino-fp16` (O1F) was measured on smoke only (same host IR
   limitation).
6. **Clean FP32 export (O0)** is an instrumented validation wrapper (37 hidden states +
   pre-norm + post-norm outputs) converted with OpenVINO's PyTorch frontend from the same
   checkpoint — not a production optimum.

## Experiment scope

7. **100-prompt statistical suite ran with 1 fixed seed (20261001) × 2 step counts × 2 arms**
   (recorded deviation from the ideal 3 seeds — bounded by single-GPU compute time;
   the canary 10-prompt suite provides the multi-seed evidence at 3 seeds where needed:
   see `suite_canary_manifest.json`). Embedding-level statistics use all 100 prompts.
8. **Native 2K production suite**: scope and seed count as recorded in
   `artifacts/metrics/suite_production_30_manifest.json` (30 prompts, 40 steps, seeds as
   recorded; deviation from 3 seeds documented there if compute-bound).
9. **Perturbation calibration** uses Gaussian random deltas with per-token norm matching
   (matches per-token scale distribution in norm, not in directionality/structure).
10. **Human blind evaluation**: package generated (`reports/blind_package/`); evaluator
    results are PENDING — no human data was fabricated. Automated metrics stand alone.
11. **PSNR/SSIM** on independently-generated images measure pixel agreement between two
    valid samples, not "correctness"; they are reported with LPIPS and OCR/CLIP-style
    checks, never alone.
12. **OCR** (RapidOCR ONNX) accuracy bounds CER/WER measurements; stylized fonts and small
    text can be under-detected equally for both arms.
13. Resource monitoring sampled at 5–20 Hz (psutil); peak-RSS attribution across arms in
    `cpu_performance.json` is cumulative within one process — per-arm load/latency numbers
    are the reliable columns.
14. Diffusion determinism was verified bitwise for the W7900D/ROCm 7.14 stack used here.
    Other driver/torch combinations may introduce non-determinism that this validation
    does not cover.
