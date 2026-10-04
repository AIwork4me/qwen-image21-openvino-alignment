# Executive Summary

## Question
Is the OpenVINO (AMD CPU) Qwen3-VL-8B text encoder for Qwen-Image 2.1 numerically aligned
with the AMD GPU (Radeon PRO W7900D, ROCm 7.14.60850) PyTorch reference — and why do
25-step images look similar while 40-step images can differ?

## Answers

1. **Alignment**: The OpenVINO FP32 implementation is numerically aligned with PyTorch FP32
   to float32 round-off (relative L2 ≈ 0.000008, cosine ≈ 1.000000).
   Conversion, runtime and operator behavior introduce NO material error.

2. **Where differences originate**: The production INT8 weight-compressed encoder adds a
   quantization-level embedding delta (rel. L2 ≈ 0.070 vs OpenVINO FP32;
   ≈ 0.097 vs the GPU BF16 reference). For scale: the GPU reference's own
   BF16-vs-FP32 precision envelope is ≈ 0.053.
   Inputs/tokenization are EXACTLY identical. The INT8 delta starts at the compressed
   embedding table (~3% at layer 0) and accumulates mildly through the trunk.

3. **Why 25 vs 40 steps differ**: Measured mechanism — the embedding delta steers the
   diffusion trajectory during the early/mid structure-forming steps; the divergence is then
   locked in (crossover experiment: swapping conditioning after step 25 changes the final
   image by only ~0.3-0.4%, while swapping the accumulated latent state changes it ~19%).
   Longer, finer 40-step schedules accumulate more divergence (final latent rel. L2:
   25 steps ≈ 7.6% vs 40 steps ≈ 18.5%; same embeddings, same initial latent, bitwise-deterministic backend).

4. **Is this an OpenVINO defect?** No. Random embedding perturbations of the SAME magnitude
   (norm-matched, unrelated to OpenVINO) produce equal or LARGER final-image differences
   (40-step: OV delta 18.5%/PSNR 18.5 vs random
   16.7-23.5%/PSNR 18.0-19.6 on S02; on C09 random deltas were 1.7-2.6x larger).
   This is generic trajectory sensitivity of Qwen-Image 2.1 to small conditioning deltas.

5. **Quality regression**: Not detected on the measured suites. Text-rendering (English and
   Chinese, OCR-verified CER) is equal between arms; perceptual/semantic metrics show no
   systematic regression (see customer report for distributions and worst cases).

6. **Production recommendation**: The OpenVINO encoder is safe to deploy for production use
   with the documented caveat: outputs are a *different valid sample* of the same prompt
   conditioning at high step counts, not a corrupted generation. Where bit-stable 40-step
   outputs vs the GPU reference are required, use FP32 or BF16 OpenVINO precision
   (BF16: rel. L2 1.3%; both visually indistinguishable in expectation) or accept
   trajectory-level variation with INT8.

## Key numbers

| Comparison | cosine (median) | rel. L2 (median) |
|---|---|---|
| ROCm repeatability (R0 vs R0R) | 1.000000 | 0.000000 |
| GPU BF16 vs FP32 (R0 vs R1) | 0.9987 | 0.053 |
| OV FP32 vs GPU FP32 (R1 vs O0) | 1.0000 | 0.000008 |
| OV runtime BF16 (O0 vs O1) | 0.9999 | 0.013 |
| OV INT8 (O0 vs O3) | 0.9976 | 0.070 |
| GPU BF16 vs OV INT8 (R0 vs O3) | 0.9957 | 0.097 |
