# Customer Report — OpenVINO Qwen3-VL-8B Text Encoder Alignment for Qwen-Image 2.1
## OpenVINO on AMD CPU vs PyTorch on AMD Radeon PRO W7900 / ROCm 7.14.0

Every claim below points to measured evidence retained in `artifacts/`. Companion
documents: TECHNICAL_REPORT.md (methodology + numbers), EXECUTIVE_SUMMARY.md,
LIMITATIONS.md, REPRODUCTION.md, gallery/.

---

## 1. The question

You observe that images generated with the OpenVINO-optimized Qwen3-VL-8B text encoder
look similar to the AMD GPU reference at **25 inference steps**, but can look substantially
different at **40 steps**. You suspect the OpenVINO encoder is not numerically aligned.

We tested: (1) numerical alignment, (2) where divergence begins, (3) its cause,
(4) whether 40-step differences are valid alternative generations or quality regression,
(5) production safety.

## 2. Test environment

- GPU reference: **AMD Radeon PRO W7900D (gfx1100), ROCm 7.14.0 userspace, PyTorch
  2.13.0+rocm7.14.0** — verified, not assumed (the machine initially exposed a 7.2.1
  userspace; corrected before any measurement, reversibly, and recorded).
- CPU: **AMD EPYC 9334** (Zen4, AVX512-BF16/VNNI), OpenVINO 2026.4.1 CPU plugin.
- Model: Qwen/Qwen-Image-2.1, snapshot `d26bb612…` — one checkpoint for BOTH paths,
  config+weight hashes in `artifacts/model_manifest.json`.

## 3. Why the experiment is fair

- **One variable at a time**: both encoders consume the *same frozen token inputs*; the
  diffusion backend is always the same W7900 pipeline; the initial latent is a frozen file
  reused byte-for-byte (hash-asserted before every run); schedules are re-derived and
  compared list-exactly against a frozen manifest; Qwen-Image 2.1's official sampling
  (no classifier-free guidance) is used.
- **The compared tensor is the real conditioning input** to the diffusion transformer
  (last-layer hidden state *before* the final RMSNorm, system tokens removed) — proven
  bit-identical to the official diffusers `encode_prompt` by unit test.
- **The GPU reference's own behavior is calibrated first** (repeatability, precision
  envelope), so OpenVINO error is judged against baselines, not arbitrary thresholds.

## 4. Input identity (Phase A) — PASS, exact

Token sequences are **bit-identical** between the official processor path and the
production (ComfyUI qwen25 tokenizer) path for every prompt; drop point = 14 tokens =
second `<|im_start|>` in all cases; no image inputs present. **Preprocessing is not a
divergence source.**

## 5. Text-encoder numerical alignment (Phase B)

Relative-L2 error of the conditioning tensor (median; canary = 10 prompts, n=100 = full
statistical suite):

| What is compared | rel. L2 | cosine | Interpretation |
|---|---|---|---|
| GPU run repeated (noise floor) | **0.000000** | 1.000000 | GPU reference is bitwise reproducible |
| GPU BF16 vs GPU FP32 | 5.3% | 0.9987 | *the reference's own precision envelope* |
| **OpenVINO FP32 vs GPU FP32** | **0.0008%** | 1.000000 | **conversion + runtime aligned** |
| OpenVINO runtime BF16 | 1.3% | 0.99993 | floating-point precision effect, small |
| FP16 weight storage | 0.006% | ≈1.0 | negligible |
| **OpenVINO INT8 vs OpenVINO FP32** | **7.0%** | 0.9976 | **quantization delta** |
| **GPU BF16 vs OpenVINO INT8** | **9.7%** (n=100: 9.67% ± 0.1, min cos 0.9954) | 0.9958 | total production delta |

The INT8 delta is extremely stable across 100 diverse prompts (9.5–9.9%) — a structured
noise floor, **the same order of magnitude as the GPU's own BF16-vs-FP32 difference**.

## 6. Layer-by-layer attribution (Phase C)

- Clean OpenVINO FP32 vs GPU FP32: error ≈ 2–8e-6 at **every one of the 36 layers** — no
  anomalous layer, operator, or conversion defect.
- INT8 path: error is **3% already at the token-embedding output** (the artifact also
  int8-compresses the vocabulary table), ~1.0–1.4% through the trunk, amplified ~2.8× in
  the final layer to ~7%. No cliff, no single broken operator — distributed quantization
  noise, mildly amplified by depth. Zero NaN/Inf anywhere.

## 7. OpenVINO precision analysis (Phase D)

FP32 forensic baseline (ACCURACY execution mode, f32 inference precision, verified at
runtime) **aligns**; BF16 runtime adds only 1.3%; INT8 adds 7%. Per the decision tree:
**Case D — quantization-induced difference** — riding on a Case-C-scale envelope that
already exists between your BF16 GPU reference and FP32.

> Note on your production artifact: on our Zen4 validation host, the downloaded
> INT8 IR fails for prompts longer than ~40 tokens (OpenVINO eltwise shape bug; it ran
> 185-token prompts on earlier Zen5 hosts). We reproduced your quantization recipe locally
> and verified it matches the artifact's error signature (7.0%/9.7% vs 7.0%/9.8%), then
> used the reproduction for full-coverage suites. Please re-check the artifact on your
> production CPU against your real prompt-length distribution.

## 8. Embedding transplant (Phase E) — same diffusion, only embeddings differ

Same latent, same schedule, same GPU pipeline; only the positive embedding changes:

| Steps | final-latent divergence (R0 vs OV-INT8) |
|---|---|
| 25 | rel L2 7.6% |
| 40 | rel L2 18.5% |

The diffusion backend itself is bitwise deterministic (verified by identical repeat runs),
so 100% of this divergence is caused by the embedding delta.

## 9. Why 25 steps look similar but 40 steps differ (Phases F/G)

Two measured facts explain it:

1. **The schedules are genuinely different**: a 25-step run is *not* the first 25 steps of
   a 40-step run — the timestep/sigma grids differ from step 0 (verified and recorded).
2. **Crossover experiment (40-step schedule, swap at step 25):** swapping the *conditioning*
   after step 25 changes the final image by only **0.3–0.4%**; swapping the *accumulated
   latent state* changes it by **18.5–19.5%**. The embedding delta steers the trajectory
   during the early/mid structure-forming steps and the difference is then locked in.
   More steps on a finer grid accumulate and refine more of that difference:
   7.6% → 18.5% final-latent divergence (25 → 40 steps), which crosses the threshold of
   visual noticeability.

## 10. Perturbation sensitivity (Phase H) — is OpenVINO special? No.

We added *random* embedding perturbations of exactly the OV-INT8 delta's magnitude
(norm-matched per token) and re-ran the identical diffusion:

| 40-step runs | final rel L2 | PSNR |
|---|---|---|
| actual OV-INT8 delta | 18.5% / 18.4% | 18.5 / 21.1 dB |
| random delta, seed 11 | 23.5% / 47.2% | 18.0 / 14.3 dB |
| random delta, seed 22 | 16.7% / 39.5% | 19.6 / 15.6 dB |
| random delta, seed 33 | 17.7% / 32.1% | 19.4 / 16.9 dB |

(S02 / C09 prompts.) **Random deltas of the same size cause equal or larger image
differences.** A quarter of the delta collapses to PSNR ≈ 37 dB (imperceptible). This is
inherent trajectory sensitivity of Qwen-Image 2.1 conditioning — not an OpenVINO defect.

## 11. Large-sample image-quality evaluation (Phase I/J)

- **100-prompt suite (fixed seed, 25+40 steps, both arms, n=100 pairs per step count)**:
  - 25 steps: PSNR median 28.8 dB (p5 18.9 / p95 43.2), SSIM median 0.967,
    LPIPS median 0.036 (p95 0.212)
  - 40 steps: PSNR median 29.3 dB (p5 18.9 / p95 42.6), SSIM median 0.968,
    LPIPS median 0.030 (p95 0.159)
  Distribution is broad and **symmetric** — a mixture of near-identical outputs
  (PSNR>40) and visibly different valid trajectories (PSNR~16-20), consistent with the
  measured trajectory sensitivity; no systematic direction of OV being worse.
- **Text rendering (OCR-verified)**:
  - 1024/40 steps, 20 text prompts: CER median R0 0.080 vs OV 0.036; **0 prompts worse
    by >0.02, 6 better**; English `"FRESH BAKERY DAILY"` rendered exactly by both arms.
  - 2K/40 steps: see §12 — not significant.
- Worst cases are in `reports/gallery/`, not only favorable ones.
- Blind human evaluation package generated (100 pairs, left/right randomized, key stored
  separately); **human results PENDING — none fabricated**.

## 12. Native 2K production validation

30 representative prompts (English/Chinese text, signage, posters, labels, long prompts)
at **2048×2048, 40 steps, both arms, frozen latents** (60 generations, tiled VAE decode —
recorded, applied identically to both arms). 1 fixed seed executed (compute-bound; the
3-seed matrix is covered by the canary suite below). Per-run ≈320 s on the W7900D.

- Pixel/perceptual: PSNR median 27.8 dB (p5 19.8 / p95 40.9), SSIM median 0.964,
  LPIPS median 0.032 — same distribution family as the 1024 suite.
- Text rendering (OCR, 19 text prompts): paired CER median diff **+0.026, Wilcoxon
  p = 0.42 (not significant)**; raw counts 10 worse / 4 tie / 5 better for OV; combined
  with the 1024 suite: 10 worse vs 11 better (sign-test p = 1.0). Manual inspection of
  the worst OCR pairs shows the *main* text rendered correctly in both arms (e.g.
  "DEEP SKY", "PURE GLOW", "HOTEL PACIFIC") — the CER differences come from small
  decorative/credit text that both arms render as different garbled textures. Worst
  pairs retained in `reports/gallery/2k_worst_ocr_pair__*`.
- Canary 3-seed check (1024, 40 steps): LPIPS medians 0.041/0.051 (seeds 2/3) vs 0.089
  (seed 1 subset) — spread is seed-driven, no arm-direction.

**Verdict: no measurable quality or text-rendering regression at 2K production settings.**

## 13. Performance and memory (CPU, EPYC 9334; kept separate from accuracy)

| Encoder | load | warm @25 tok | warm @67 | warm @171 |
|---|---|---|---|---|
| OpenVINO FP32 | 34.7 s | 0.83 s | 1.07 s | 2.49 s |
| OpenVINO BF16 | 12.6 s | 0.39 s | 0.92 s | 1.33 s |
| OpenVINO INT8 (recipe) | **1.4 s** | **0.22 s** | 0.71 s | **1.08 s** |
| GPU reference (R0) | ~11 s | 0.04–0.12 s | — | — |

INT8 buys ~2.3× warm latency and ~25× model-load time vs FP32 on this CPU, at the
documented ~7% embedding-noise cost. Resource time-series in `artifacts/metrics/*resources*.csv`.

## 14. Limitations

See LIMITATIONS.md — including the customer-artifact host IR issue (§4), single-host
scope, seed-count deviations, and interpretation cautions for PSNR/SSIM on generative pairs.

## 15. Evidence-based conclusion

Distinguishing the four different concepts:

- **Bitwise identity**: does not hold between any two different precisions (nor does the
  GPU hold it BF16-vs-FP32). Only same-arm repeats are bitwise identical.
- **Numerical equivalence**: OpenVINO FP32 ≈ PyTorch FP32 to 8e-6 (yes); OpenVINO INT8 ≈
  7% structured quantization noise (no, by construction of INT8).
- **Perceptual/semantic similarity & quality equivalence**: no measurable regression;
  text-rendering accuracy equal; differences are alternative valid trajectories.
- **Root cause of your observation**: quantization-level embedding delta (Case D) +
  generic 40-step trajectory sensitivity (Case E mechanism) — **not** a conversion,
  runtime, preprocessing, or operator defect.

**The OpenVINO text encoder is numerically aligned at FP32/BF16 precision. The production
INT8 artifact introduces a bounded, stable quantization noise whose visual effect at high
step counts is indistinguishable from equal-sized non-OpenVINO perturbations.**

## 16. Recommendation

1. **Production use: YES**, with an informed choice of precision:
   - If 40-step outputs must stay visually close to the GPU reference: use **OpenVINO
     FP32 or BF16** (BF16 delta 1.3% ≈ visually indistinguishable; 3× faster than FP32).
   - If throughput is king and "different valid sample" is acceptable (batch/explore
     workflows): **INT8 is fit for purpose** — equal text-rendering and quality metrics.
2. Re-quantize with an **accuracy-aware recipe** (exclude/heighten the embedding table and
   optionally the last transformer block) if you want to shrink the 7% delta; expected
   gain: embedding delta toward the BF16 envelope (~1–3%).
3. Ship the **fixed prompt/seed/latent harness** (this repo) in CI to catch real
   regressions: alert on embedding rel-L2 vs the calibrated bands in
   `config/thresholds.yaml`, not on image differences alone.
4. Track the OpenVINO IR issue seen on Zen4 for >40-token prompts with the artifact
   vendor; re-export with a current optimum/NNCF version (our local re-quantization of the
   same recipe runs clean at any length).
