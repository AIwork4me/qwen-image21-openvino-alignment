# ComfyUI `qwen3vl_8b_w4a8.safetensors` — Format Specification

> v2 reverse-engineering deliverable. Evidence classes: **[CHECKPOINT]** direct
> artifact inspection, **[SOURCE]** ComfyUI/comfy-kitchen source at frozen revision,
> **[INFERENCE]** derived, **[UNKNOWN]** unverified.
> Artifact SHA256 `7754425e55e7bea2bfde4dde59a4cc236cb44f5ee9c215ea66ef8d47012824eb`.

## 1. Summary

**"W4A8" = 4-bit weight codes + Lloyd-Max codebook + per-group FP8 relative scales +
per-channel fp32 scale, all in the ConvRot Hadamard-rotated basis (group 256), designed
to execute through the shared INT8 GEMM ("A8" = activations quantized to INT8 in the
fast-kernel path).** Two sensitive tensors (`model.embed_tokens`, `lm_head`) are stored
as INT8 ConvRot (W8), not W4. **[CHECKPOINT + SOURCE]**

## 2. Container layout **[CHECKPOINT]**

- File: 6,312,105,364 bytes, 1,762 tensors, safetensors metadata `{"format": "pt"}`.
- 252 W4 layers (all 36 blocks × 7 projections), each stored as:
  - `weight` : I8 `[N, K/2]` — two 4-bit codes per byte (even column = low nibble)
  - `weight_s_rel` : F8_E4M3 `[N, K/16]` — per-group relative scale (group_size = 16)
  - `weight_s_channel` : F32 `[N]` — per-output-channel absolute scale
  - `weight_codebook` : F32 `[16]` — per-layer Lloyd-Max codebook (symmetric)
  - `weight_correction` : **absent** for all layers ⇒ symmetric variant **[CHECKPOINT]**
  - `comfy_quant` : `{"convrot": true, "convrot_groupsize": 256, "format": "asym_w4a8_int8", "group_size": 16}`
- `model.embed_tokens` and `lm_head`: `int8_tensorwise` + convrot (identical to the
  INT8 ConvRot artifact format — I8 `[151936, 4096]` + F32 `[151936, 1]` scale).
- Norms / biases stay BF16 (496 tensors, same set as the INT8 artifact).

## 3. Exact encode/decode math **[SOURCE]**

Quantization (`quantize_w4a8_int8_weight`, eager reference):

1. Rotate offline: `W_rot = W_blk @ H^T` (Hadamard-256 groups, as INT8 ConvRot §3).
2. Reshape to groups of 16. Initial `group_scale = absmax(group)` (clamped ≥ 1e-8);
   `normalized = group / group_scale`.
3. Codebook: fixed 16-level Lloyd-Max table for a unit-variance Gaussian
   (`_FIXED_LUT`, values ±0.05…±0.98) if the sampled excess kurtosis of normalized
   values ≤ −0.1 (rotated groups are Gaussian in practice), else per-tensor k-means fit.
   One codebook is shared by the whole tensor (strided-row sample decides).
4. Level assignment: nearest code (monotone searchsorted; ties → lower index).
5. **ALS refinement ×2**: `group_scale = Σ(w·level)/Σ(level²)` then reassign levels.
6. Reconstruct `shifted = level × group_scale`; per-channel scale
   `s_channel[n] = absmax_n(shifted)/127`.
7. `s_rel[n,g] = group_scale/s_channel` stored as **FP8 e4m3** (rounding of the stored
   scale is part of the format).
8. **Final grid assignment**: codes are re-decided against the INT8 grid:
   `levels_g = round(clamp(LUT[c] × s_rel_fp8)) ∈ int8`, i.e. the decoded weight is an
   exact INT8 value per element. `unsigned = code index` (0..15), packed 2/byte.

Dequantization (effective weight actually multiplied during inference):

```
W_rot_eff[n,k] = round( clamp( LUT[code[n,k]] × s_rel_fp8[n, k//16], -127, 127 ) ) × s_channel[n]
W_eff = unrotate( W_rot_eff )          # per Hadamard-256 group: W_blk @ H
```

**The double rounding (s_rel in fp8, then product onto the int8 grid) is part of the
format and must be reproduced exactly for bit-level weight parity.** **[SOURCE]**

## 4. Runtime execution path **[SOURCE + profile]**

- Same loader chain as INT8 (format string `asym_w4a8_int8` → `AsymW4A8Int8Layout`).
- Stock conditioning path (`encode_from_tokens`, `full_precision_mm=True`):
  each W4 layer **dequantizes to its effective weight via the exact formula above
  and runs a regular matmul** — with the **FP32 activation stream** that `sd1_clip`
  forces (`out_dtype`/`dtype=torch.float32`). The "A8" INT8 activation quantization
  does NOT execute in the path that produces Qwen-Image 2.1 conditioning. Evidence:
  committed profiler traces `artifacts/v2/metrics/stock_path_profile_C_W4A8.json`
  (`dequantize_w4a8_int8_weight` per layer per encode, no W4A8 GEMM kernel).
- Fast-kernel path (`--fast-kernels` / `use_quantized_matmul`, used by `CLIP.generate`
  and diffusion models): codes decoded to the INT8 grid in-kernel, activation rotated +
  per-row INT8-quantized, shared INT8 GEMM, `s_channel` epilogue (`w4a8_int8_linear`).

## 5. Distinguishing W4A8 from INT8 ConvRot **[CHECKPOINT]**

| Property | INT8 ConvRot | W4A8 |
|---|---|---|
| Weight codes | INT8 per-row symmetric | INT4 + Lloyd-Max codebook per group |
| Scales | F32 per-channel | FP8-E4M3 per-16-group relative + F32 per-channel |
| Codebook | none | F32[16] per layer |
| Rotation | Hadamard-256 offline | Hadamard-256 offline (same) |
| embed/lm_head | INT8 ConvRot | INT8 ConvRot (identical treatment) |
| Norms/biases | BF16 (vision tower BF16, unquantized) | BF16 (same) |
| Stock-path activation dtype | FP32 activation stream (weight-only dequant + mm) | FP32 activation stream (same) |
| Fast-path activation dtype | INT8 per-row dynamic | INT8 per-row dynamic |

## 6. Open questions **[UNKNOWN]**

- Whether the official producer used the fixed Gaussian LUT or a fitted codebook per
  tensor (decidable from the checkpoint: compare stored `weight_codebook` against
  `_FIXED_LUT` — performed in the weight-parity experiment).
- Stochastic rounding seeds / calibration provenance (not needed for decode parity).
- Whether `scale_search` (grid-aware scale refinement, W6A8-only in source) was ever
  applied to 4-bit production (source gates it to bits=6 — so no). **[SOURCE]**
