# Technical Report — Qwen-Image 2.1 Text Encoder Alignment Validation
## OpenVINO (AMD CPU) vs PyTorch (AMD Radeon PRO W7900D, ROCm 7.14.0)

All numbers in this report come from raw artifacts under `artifacts/` (tensors `.npz`,
metrics `.csv/.json`, logs). Regenerate with `scripts/build_report.py` and the commands in
`REPRODUCTION.md`.

---

## 1. Environment (measured, not assumed)

| Item | Value | Evidence |
|---|---|---|
| CPU | AMD EPYC 9334 (Zen4, 2×32C, AVX512-BF16/VNNI) | `environment/lscpu.txt` |
| GPU | AMD Radeon PRO W7900D, **gfx1100**, 48 GiB | `environment/summary.json`, `rocminfo.txt` |
| ROCm userspace | **7.14.60850** (`/opt/rocm -> /opt/rocm-7.14.0`) | `rocm_symlink.txt`, `rocm_hip_version_header.txt` |
| PyTorch | 2.13.0+rocm7.14.0 (hip 7.14.60850) | `torch_check.txt` |
| OpenVINO | 2026.4.1-22982 | `summary.json` |
| transformers / diffusers | 5.18.0 / 0.40.0 + vendored qwenimage21 @ 8b33bfc | `pip_freeze.txt` |
| Model | Qwen/Qwen-Image-2.1 @ d26bb612 (TE 36L, d=4096, GQA 32/8, vocab 151936) | `model_manifest.json` |

ROCm correction: system originally had 7.2.1 userspace under a 7.14-built torch; corrected
via therock 7.14.0 dist + symlink flip; GPU compute re-verified before any measurement.

## 2. Scientific design

- **Single-variable rule**: the text-encoder implementation is the only changed component.
  Diffusion backend is always the same W7900/ROCm pipeline; initial latents are frozen
  safetensors reused byte-for-byte (hash-asserted); schedules are re-derived and
  list-compared against the frozen manifest; Qwen-Image 2.1 samples without CFG
  (`true_cfg_scale=1.0`, no negative prompt — official default), one transformer call/step.
- **Arms**: R0 (ROCm BF16), R0R (repeat), R1 (ROCm FP32), O0 (clean OV FP32,
  `EXECUTION_MODE_HINT=ACCURACY`, `INFERENCE_PRECISION_HINT=f32`), O1 (O0 @ bf16 runtime),
  O1F (fp16-storage artifact), O3 (customer INT8 artifact), O3r (locally reproduced INT8
  recipe: NNCF `INT8_ASYM, ratio=1.0, group_size=-1` + row-wise int8-asym embedding table).
- **Official conditioning semantics**: T2I template
  `"<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"`,
  processor left-padding, `hidden_states[-1]` **before final RMSNorm** (norm forward-hook
  neutralization exactly as the official pipeline does under transformers 5.x), drop first
  14 tokens (tokenized system turn == second `<|im_start|>`), zero right-padding + mask.
  Unit tests in `tests/test_conditioning_semantics.py` prove bit-parity of the frozen R0
  embeddings with the official `QwenImage21Pipeline.encode_prompt`.

## 3. Phase A — input identity

- Processor path vs customer's ComfyUI `qwen25_tokenizer` path: `input_ids` EXACT for all
  prompts (13/13 smoke+canary, and by construction for all suites since both encoder
  families consume the same frozen `artifacts/inputs/*.npz`).
- Text-only inputs confirmed: no `pixel_values`, no `image_grid_thw`; `mm_token_type_ids`
  all zeros. `drop_idx=14` equals the second `<|im_start|>` position for every prompt.
- **Verdict: preprocessing is not a divergence source (Case A excluded).**

## 4. Phase B — final embedding alignment

Reference distributions (prenorm tensor, the conditioning input):

| Pair | cosine (median, min) | rel L2 (median) | bf16-exact |
|---|---|---|---|
| R0 vs R0R (repeatability) | 1.000000 / 1.000000 | **0.000000** | 1.000 |
| R0 vs R1 (GPU precision envelope) | 0.998711 / 0.998597 | 0.0528 | 0.080 |
| **R1 vs O0 (conversion/runtime)** | **1.000000 / 0.99999999** | **7.7e-06** | 0.995 |
| R0 vs O0 | 0.998711 | 0.0528 | 0.080 |
| O0 vs O1 (runtime bf16) | 0.999928 | 0.0128 | 0.141 |
| O0 vs O1F (fp16 storage) | ≈1.0 | 5.7e-05 | 0.962 |
| O0 vs O3 (customer INT8, smoke) | 0.997622 | 0.0698 | 0.031 |
| **O0 vs O3r (INT8 recipe, n=100)** | **0.997623 / 0.997449** | **0.0696 (max 0.0721)** | 0.031 |
| R0 vs O3 (smoke) | 0.995805 | 0.0974 | 0.029 |
| **R0 vs O3r (n=100)** | **0.995739 / 0.995390** | **0.0967 (max 0.0991)** | 0.028 |

100-prompt distribution is extremely tight (rel L2 9.5–9.9%): the INT8 delta is a stable,
structured noise floor, not prompt-dependent failure.

Chain conclusion: conversion OK (1e-5) → runtime bf16 OK (1.3%) → **the entire R0↔O3 gap is
the INT8 quantization delta riding on top of the GPU's own bf16 precision envelope**
(5.3%). **Case B and C excluded; Case D (quantization) confirmed as the only material
numerical mechanism.**

## 5. Phase C — 36-layer attribution

- R1 vs O0: rel L2 ≈ 1.8e-6 … 7.9e-6 at every depth (uniform fp32 round-off). No layer
  anomaly → conversion/operator behavior clean.
- O0 vs O3(r): error **starts at the embedding table** (3.0e-2 at entry 0 — the artifact
  int8-compresses `embed_tokens`), stays 1.0–1.4e-2 through layers 1–34, rises to 2.5e-2 at
  layer 34→35 output and ~7e-2 at the final (last-layer amplification ≈ 2.8×, no cliff,
  no broken operator). NaN/Inf: zero everywhere.
- R0 vs R1 grows smoothly 0.4%→5.3% (bf16 envelope shape).

## 6. Phase D — OpenVINO forensic baseline & conversion audit

- O0 compiled with `EXECUTION_MODE_HINT=ACCURACY` + `INFERENCE_PRECISION_HINT=f32`
  (verified via runtime properties; `PERFORMANCE_HINT=ACCURACY` is not a legal value in
  OpenVINO 2026.x — accuracy semantics live in EXECUTION_MODE — recorded).
- Conversion: PyTorch frontend (`ov.convert_model`), traced from fp32 weights,
  `compress_to_fp16=False`, dynamic sequence dims [1,2048]. The traced embedding submodel
  hits an unsupported `aten::embedding` Convert; token embedding is done as a bit-exact
  fp32 gather from the same checkpoint (integer-index lookup).
- Customer artifact provenance: `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` style export
  (optimum 2.1.0.dev0, INT8 default quant, processor Qwen/Qwen3-VL-8B-Instruct), full
  hashes in `artifacts/model_manifest.json` and `manifests/openvino_conversion_manifest.json`.
- Host IR bug documented (see LIMITATIONS §4): artifact fails >40 tokens on this Zen4 host.

## 7. Phase E — embedding transplant (single-variable)

Same frozen latent (sha-asserted), same timesteps (list-equal to manifest), same pipeline:
- 25 steps: final latent cos 0.9971 / rel L2 7.59%
- 40 steps: final latent cos 0.9829 / rel L2 18.49%
Backend self-repeat: bitwise identical (md5-identical PNGs) → all divergence is
embedding-caused.

## 8. Phase F — trajectory tracing

Stepwise latent divergence grows monotonically through the schedule for both 25- and
40-step runs (see `metrics/trace_S02_*.json`, plots `diffusion_*.png`). The 40-step
schedule is NOT the 25-step schedule plus 15 steps: timesteps/sigmas differ from step 0
(linspace grids 1→1/25 vs 1→1/40 with resolution-dependent mu shift; recorded in
`latents/schedule_overlap_check.json`: overlap equality = False).

## 9. Phase G — 25→40 crossover (causal localization)

40-step schedule, crossover at step k=25, prompts S02 and C09:

| pair | S02 rel L2 | C09 rel L2 |
|---|---|---|
| A(R0-lat,R0-cond) vs B(R0-lat,OV-cond) | **0.0033** | **0.0041** |
| A vs C(OV-lat,R0-cond) | 0.1852 | 0.1948 |
| A vs D(OV-lat,OV-cond) | 0.1852 | 0.1949 |
| C vs D | 0.0033 | 0.0041 |

**Interpretation (corrected after independent review):** the final image is determined by
the accumulated latent state at step 25 — swapping conditioning afterwards changes ~0.3–0.4%.
The customer-visible 40-step divergence is therefore **accumulated trajectory divergence
built during the early/mid (structure-forming) steps** under the persistent embedding delta;
late-step conditioning sensitivity is negligible. (Crossover runs disable the KV cache for
soundness of mid-run swaps; verified to reproduce the cached pipeline's final divergence
A|D = 0.18516 vs transplant 0.18490.)

## 10. Phase H — perturbation sensitivity

delta = E_OV − E_ROCM per prompt; variants E0+{0.25,0.5,1,2}·delta and norm-matched random
deltas (3 seeds), same latent/steps:

S02 (40 steps): OV-delta → rel L2 18.5%, PSNR 18.5 dB; random → 23.5/16.7/17.7%,
PSNR 18.0/19.6/19.4 dB.
C09 (40 steps): OV-delta → 18.4%, PSNR 21.1; random → 47.2/39.5/32.1%, PSNR 14.3/15.6/16.9.

**The OV-INT8 delta's image-level effect is within (and on C09 below) the random-perturbation
band of equal magnitude → generic trajectory sensitivity, not an OpenVINO-specific defect
(Case E mechanism confirmed at the image level; Case F requires a quality regression —
none found, §11).** 0.25×delta at 40 steps: rel L2 1.8% / PSNR 37 dB — below perception.

## 11. Phases I/J — image quality, text rendering, suites

**100-prompt suite (n=100 pairs per step count, fixed seed 20261001, both arms):**

| steps | PSNR median (p5/p95) | SSIM median | LPIPS median (p5/p95) |
|---|---|---|---|
| 25 | 28.78 (18.9/43.2) | 0.967 | 0.036 (0.004/0.212) |
| 40 | 29.29 (18.9/42.6) | 0.968 | 0.030 (0.003/0.159) |

Canary 3-seed check (1024, 40 steps): LPIPS medians 0.041/0.051 (seeds 20261002/03) —
spread is seed-driven; no arm direction.

**Text rendering (OCR = RapidOCR ONNX; paired stats):**
- 1024/40 steps, 20 text prompts: CER median R0 0.080 vs O3r 0.036; **0 prompts worse by
  >0.02, 6 better** (OV ≥ reference on every prompt at this resolution).
- 2K/40 steps (§12): paired CER median diff +0.026, Wilcoxon p=0.42 (not significant);
  raw 10 worse/4 tie/5 better; combined sign test across resolutions p=1.0. Worst pairs
  inspected: main text rendered by both arms; CER differences driven by OCR on small
  decorative text (e.g. poster credits).

**Native 2K suite**: 30 prompts × 40 steps × both arms @2048² (60 gens, ≈320 s each,
tiled VAE decode — recorded change, applied identically to both arms): PSNR median
27.85 (p5 19.8/p95 40.9), SSIM 0.964, LPIPS 0.032. Same distribution family as 1024.

**Blind evaluation**: 100-pair package generated, randomized (49/51 L/R split), key
stored separately; human results PENDING (none fabricated).

Key figures: `artifacts/plots/` — precision_chain.png, layer_*_vs_depth_*.png,
embedding_histogram/heatmap, diffusion_latent_divergence_{25,40}steps.png,
diffusion_model_output_*.png, steps_25_vs_40_comparison.png,
perturbation_vs_final_divergence.png, suite100_quality_distributions.png,
text_rendering_accuracy.png, ov_accuracy_performance_tradeoff.png.

## 12. Performance (CPU, EPYC 9334, accuracy configs, separate from accuracy conclusions)

| Arm (EXECUTION_MODE_HINT=PERFORMANCE) | load (read+compile) | warm p50 @25 tok | @67 | @171 | p95 @171 |
|---|---|---|---|---|---|
| O0 FP32 | 34.7 s | 0.834 s | 1.074 s | 2.492 s | 2.516 s |
| O1 BF16 | 12.6 s | 0.387 s | 0.919 s | 1.330 s | 1.372 s |
| O3r INT8 | **1.41 s** | **0.218 s** | 0.711 s | **1.080 s** | 1.208 s |

GPU R0 reference encode: 0.043–0.118 s warm (short prompts). Customer artifact O3 could not
be timed on this host (IR bug). INT8 accuracy cost (rel L2 ~7% vs OV FP32) buys ~2.3×
warm latency and ~25× load time vs FP32 on this CPU.

## 13. Failure log (never converted to PASS)

- O1F/O3 artifact IR eltwise failure >40 tokens on this host (documented, worked around
  via O3r reproduction).
- Traced embeddings submodel unsupported Convert (replaced by exact numpy gather).
- Initial `/workspace` disk exhaustion → cache relocated to `/valdata` (recorded).
- `PERFORMANCE_HINT=ACCURACY` invalid in OV 2026.x → EXECUTION_MODE used.
- One O1F canary batch and one O3 canary batch failed (host IR bug) — marked missing,
  never silently skipped.

## 14. Raw artifact map

- tensors: `artifacts/tensors/<arm>/<pid>.npz` (hidden_states[37,seq,4096], prenorm,
  postnorm, prompt_embeds, bf16 view, drop_idx; hashes in `meta_*_*.json`)
- traces: `artifacts/metrics/trace_*.json`, `tensors/traces/*.npz`
- images: `artifacts/images/<suite>_<pid>_s<seed>_<steps>s_<arm>.png`
- metrics: `artifacts/metrics/*.csv|json` (all comparison tables)
- plots: `artifacts/plots/*.png` (embedding histograms/heatmap, layer curves)
- reviews: `artifacts/reviews/*.md`
- integrity: `python scripts/verify_artifacts.py`
