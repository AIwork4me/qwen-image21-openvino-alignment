# ComfyUI INT8 ConvRot Format Specification (`qwen3vl_8b_int8_convrot.safetensors`)

Status labels used throughout: `[S]` confirmed from source, `[M]` confirmed from
artifact metadata/tensors, `[I]` experimentally inferred, `[U]` unknown.

Artifact under validation (frozen 2026-10-05, this machine):

```
repo:   https://huggingface.co/Comfy-Org/Qwen-Image-2.1
path:   text_encoders/qwen3vl_8b_int8_convrot.safetensors
size:   9,350,798,360 bytes
sha256: 8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f
tensors: 1258
```

(SHA256 independently reproduces the value recorded by the prior v2 attempt on
a different host — artifact is byte-identical cross-machine.)

## 1. What is quantized

`[M]` Every language-model matrix of Qwen3-VL-8B-Instruct is INT8:

- 36 decoder layers × 7 projections: `self_attn.{q,k,v,o}_proj`,
  `mlp.{gate,up,down}_proj`
- `model.embed_tokens` (vocabulary embedding table)
- `lm_head`

`[M]` Kept in BF16: all RMSNorm weights (`input_layernorm`,
`post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm`), the entire
`model.visual` tower (irrelevant for text-only T2I conditioning but present in
the file). The language model has no linear biases.

## 2. Per-layer storage

`[M]` For every quantized module the safetensors file carries:

| key | dtype | shape | meaning |
|---|---|---|---|
| `<mod>.weight` | I8 | `[N, K]` | rotated, row-wise-quantized weight |
| `<mod>.weight_scale` | F32 | `[N, 1]` | per-output-channel scale (of the rotated weight) |
| `<mod>.comfy_quant` | U8 | `[72]` | JSON config string |

`[M]` `comfy_quant` payload for every layer (embed, lm_head, all linears):

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

Example shapes `[M]`: `q_proj` I8 `[4096, 4096]`, `gate_proj` I8 `[12288, 4096]`,
`embed_tokens` I8 `[151936, 4096]` with scale `[151936, 1]`.

## 3. ConvRot: the rotation

`[S]` (comfy_kitchen `tensor/int8_utils.py`):

- `H` = regular orthogonal Hadamard matrix of size 256, built as the 4-fold
  Kronecker power of the fixed base
  `h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, normalized by
  `sqrt(256) = 16`. Deterministic, parameter-free, identical everywhere
  (no rotation matrix is stored in the file).
- `h4` is symmetric and orthogonal ⇒ `H = H^T = H^-1`: applying the rotation
  twice returns the original basis.
- Weight rotation (offline, at quantization time):
  `W_rot = W @ H^T` applied independently to each contiguous block of 256
  input columns (`W.reshape(N, K/256, 256) @ H^T`).
- Activation rotation (online, at inference):
  `x_rot = x @ H` per 256-feature group, fused into the GEMM path.

## 4. Quantization / dequantization math

`[S]` Quantization (per output channel n, over the rotated row):

```
scale[n]   = absmax(W_rot[n, :]) / 127        (fp32, floor 1e-30)
q[n, k]    = round_int8(W_rot[n, k] / scale[n])
```

`[S]` Dequantization (exact inverse path used for weight-space parity):

```
W_rot_hat[n, k] = q[n, k] * scale[n]          (fp32)
W_hat          = W_rot_hat @ H^T              (un-rotate, same H)
```

`[S]` Linear computation at inference (`int8_linear` eager reference):

```
x_rot  = x @ H                     (bf16/fp32, per 256-group)
x_q, xs = rowwise_int8(x_rot)      (dynamic per-row: xs = absmax/127)
acc    = x_q @ q^T                 (int32 accumulation)
y      = acc * xs[:, None] * scale[None, :] + bias   (fp32, then cast to bf16)
```

So INT8 ConvRot is **weight int8 + dynamic activation int8 (W8A8)** in a
Hadamard-rotated basis; the rotation is exactly invertible and adds no
approximation by itself — all quantization error comes from the int8 grids.

## 5. Embedding path

`[S]` Embedding lookup uses `dequantize_int8_embedding`: gathers int8 rows,
multiplies per-row scale, un-rotates each row (`@ H^T`) in fp32. (A linear
that consumes the whole table folds the un-rotation into its GEMM instead.)

## 6. Kernel paths

`[S]` Backends: `torch.ops.comfy_kitchen.int8_linear` dispatches to
CUDA/HIP/Triton/Ascend/eager implementations; on this W7900/gfx1100 stack the
HIP backend serves the GEMM (`_int8_matmul_accumulate` → `torch._int_mm` in
the eager fallback). Layer-norm-weight dtype (BF16) sets compute dtype.

## 7. Verification hooks in this repo

- `artifacts/v2/manifests/tensor_index_c_int8.json` — full tensor index `[M]`
- `scripts/v2_weight_parity.py` — dequantizes every tensor via the exact
  comfy_kitchen eager reference and compares to `qwen3vl_8b_bf16.safetensors`
  (`artifacts/v2/metrics/int8_weight_parity.csv`)
