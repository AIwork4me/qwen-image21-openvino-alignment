# ComfyUI `qwen3vl_8b_int8_convrot.safetensors` — Format Specification

> v2 reverse-engineering deliverable. Every statement below is tagged with its
> evidence class: **[CHECKPOINT]** direct inspection of the frozen artifact,
> **[SOURCE]** ComfyUI / comfy-kitchen source at the frozen revision,
> **[INFERENCE]** experimentally or logically derived, **[UNKNOWN]** not verified.
> Artifact SHA256 `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f`
> (see `artifacts/v2/manifests/comfyui_encoder_artifacts.json`).

## 1. What "INT8 ConvRot" means

**INT8 ConvRot = INT8 weights with an offline group-wise orthogonal Hadamard
rotation ("ConvRot", group size 256) applied to the weight basis, plus per-output-channel
(per-row) scaling.** It is a W8 quantization in a rotated basis; in exact arithmetic the
rotation is mathematically transparent (`H` is orthogonal), and exists to flatten activation
outliers so that low-bit per-row scaling stays accurate. The name derives from
comfy-kitchen's `int8_convrot` kernel family; "Conv" refers to the convergence-friendly
Hadamard construction (Kronecker powers of the regular 4×4 Hadamard), "Rot" to the rotation.
**[SOURCE]**

The concept matches the published QuaRot/SpinQuant family of Hadamard rotations
(prior-art context, not required for correctness). **[INFERENCE]**

## 2. Container layout **[CHECKPOINT]**

- File: `qwen3vl_8b_int8_convrot.safetensors`, 9,350,798,360 bytes, 1,258 tensors.
- safetensors metadata: `{"format": "pt"}` only (no `_quantization_metadata` blob;
  quantization is described per-layer by `.comfy_quant` tensors — "version 1" style).
- 254 quantized modules, each stored as:
  - `<layer>.weight` : I8 `[N, K]` — rotated, per-row-quantized INT8 codes
  - `<layer>.weight_scale` : F32 `[N, 1]` — per-output-channel scale (NOT scalar;
    the format string says "tensorwise" but convrot implies per-channel scales — see §4)
  - `<layer>.comfy_quant` : U8 (JSON bytes) — for all 254 layers identically:
    `{"convrot": true, "convrot_groupsize": 256, "format": "int8_tensorwise"}`
- 496 BF16 tensors remain: all RMSNorm weights (input/post_attention layernorm,
  q_norm, k_norm — 4+2 per block ×36, `model.norm`), attention biases
  (q/k/v/o `bias` — Qwen3-VL attention QKV bias is retained), and equivalent keys.
- Quantized module list (complete): `model.embed_tokens`, `lm_head`, and for each of
  the 36 blocks `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` (7 × 36 = 252).
  **No layer is excluded**; the vision tower is absent (text-only artifact).

## 3. Exact quantization / dequantization math **[SOURCE]**

Let `W ∈ R^{N×K}` be the original bf16 weight, `g = 256` (convrot_groupsize),
`K` divisible by `g` (4096, 1024, 4304, 151936-row embeddings with K=4096 all qualify).

1. **Hadamard**: `H = kron^m(H4)/sqrt(g)`, `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`,
   `g = 4^(m+1)`; for g=256, `m=3`. `H` is symmetric and orthogonal
   (`comfy_kitchen/tensor/int8_utils.py::_build_hadamard`).
2. **Offline weight rotation**: reshape `W -> [N, K/g, g]`; `W_rot = W_blk @ H^T`;
   flatten back (eager `_rotate_weight`; GPU: `quantize_int8_convrot_weight`).
3. **Per-row INT8**: `s[n] = absmax(W_rot[n,:]) / 127` (clamped ≥ 1e-30),
   `q[n,:] = round(W_rot[n,:] / s[n])` (round-to-nearest; stochastic rounding optional
   at quantize time, not used in this artifact's decode path).
4. **Dequantization** (exact inverse used by ComfyUI when running the layer):
   `W_eff = unrotate(round(q × s)) = (q.float() × s) @ H` per group
   (`dequantize_int8_convrot_weight`).
5. **Embedding table**: quantized identically (per-vocab-row scale, rotated columns);
   lookup gathers int8 rows, scales, then un-rotates only the gathered rows
   (`dequantize_int8_embedding`, group_size folded into the kernel).

## 4. Runtime execution path in ComfyUI **[SOURCE]**

- Loader: `comfy.sd.load_clip(..., CLIPType.QWEN_IMAGE)` →
  `load_torch_file` → `convert_old_quants` (keeps `.comfy_quant` keys) →
  `llama_detect` (`detect_layer_quantization` finds `.comfy_quant` → `mixed_ops`) →
  `qwen_image21.te(llama_quantization_metadata=...)` →
  `sd1_clip` builds `comfy.ops.mixed_precision_ops(..., full_precision_mm=True)`.
- **Conditioning encode path** (`CLIP.encode_from_tokens`, used by
  CLIPTextEncode for Qwen-Image 2.1): `_full_precision_mm = True` ⇒ every quantized
  Linear/Embedding **dequantizes to bf16 and runs a regular matmul** — i.e. the
  stock ComfyUI conditioning path executes INT8 ConvRot as *weight-only quantization
  with bf16 activations*. The dynamic per-row INT8 **activation** quantization
  ("A8") exists only behind `comfy.ops.use_quantized_matmul(...)`, which
  `encode_from_tokens` does NOT enter (only `CLIP.generate` and diffusion-model
  paths do). Verified by profiler trace: 252 × `dequantize_int8_convrot_weight_dtype`
  + bf16 `aten::mm` per encode, zero int8 GEMM kernels. **[CHECKPOINT via profile]**
- Fast-kernel path (supplementary arm `C_INT8` `--fast-kernels`): activation rotated
  online `x_rot = x_blk @ H`, per-row dynamic INT8 quant, INT8×INT8 GEMM with fp32
  accum, scales applied in the epilogue (`int8_linear(convrot=True, g=256)`).

## 5. Numerical expectations **[INFERENCE]**

- Weight-space: `W_eff` differs from `W` by INT8 per-row quantization noise in the
  rotated basis (measured in `int8_weight_parity.csv`).
- Execution in the stock path: error ≈ weight-dequantization error only (no activation
  quantization), computed in bf16.
- Embedding row exactness: the embedding lookup is exact dequant of quantized rows
  (no rotation error in exact arithmetic; H @ H^T = I).

## 6. Open questions **[UNKNOWN]**

- Which exact producer script quantized the official artifact (scale rounding mode,
  calibration data, stochastic rounding seed) is not public; irrelevant for decode
  parity since scales are stored explicitly.
- Whether `weight_scale` was rounded to fp32 after per-row absmax in bf16 or fp32
  (fp32 storage suggests fp32 computation; decode uses stored values either way).
