# ComfyUI W4A8 Format Specification (`qwen3vl_8b_w4a8.safetensors`)

Status labels: `[S]` confirmed from source, `[M]` confirmed from artifact
metadata/tensors, `[I]` experimentally inferred, `[U]` unknown.

Artifact under validation (frozen 2026-10-05, this machine):

```
repo:   https://huggingface.co/Comfy-Org/Qwen-Image-2.1
path:   text_encoders/qwen3vl_8b_w4a8.safetensors
size:   6,312,105,364 bytes
sha256: 7754425e55e7bea2bfde4dde59a4cc236cb44e5ee9c215ea66ef8d47012824eb
tensors: 1762
```

## 1. What is quantized

`[M]` Language-model decoder linears (36 layers × 7: `self_attn.{q,k,v,o}_proj`,
`mlp.{gate,up,down}_proj`) use **W4A8** (4-bit non-uniform weight codes,
8-bit dynamic activations) in the ConvRot Hadamard-256 rotated basis.

`[M]` `model.embed_tokens` and `lm_head` use **INT8 ConvRot** — the identical
per-channel int8 + Hadamard scheme as `qwen3vl_8b_int8_convrot.safetensors`
(`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`,
weight I8 `[151936, 4096]`, scale F32 `[151936, 1]`).

`[M]` BF16: all RMSNorm weights and the whole `model.visual` tower.

## 2. Per-linear storage

`[M]` `comfy_quant` payload for every decoder linear:

```json
{"format": "asym_w4a8_int8", "group_size": 16, "convrot": true, "convrot_groupsize": 256}
```

| key | dtype | shape | meaning |
|---|---|---|---|
| `<mod>.weight` | I8 | `[N, K/2]` | packed 4-bit codes, 2 per byte |
| `<mod>.weight_s_rel` | F8_E4M3 | `[N, K/16]` | per-group relative scale |
| `<mod>.weight_s_channel` | F32 | `[N]` | per-output-channel scale |
| `<mod>.weight_codebook` | F32 | `[16]` | shared 16-level Lloyd-Max codebook |

Example `[M]`: `q_proj` I8 `[4096, 2048]` (K=4096), `down_proj` I8
`[4096, 6144]` (K=12288), `s_rel` `[4096, 768]`, codebook `[16]`.

`[S]` Nibble packing (`_pack_codes`/`_unpack_codes`): even column holds the
low nibble, odd column the high nibble of each byte (`codes[:,0::2] & 0xF |
(codes[:,1::2] & 0xF) << 4`); decode reads the byte as uint8 (no sign
extension).

## 3. Quantization pipeline (how the artifact was produced)

`[S]` (comfy_kitchen `backends/eager/w4a8_int8.py`):

1. Rotate weight into Hadamard-256 basis: `W_rot = W @ H^T` (identical
   ConvRot H as the INT8 format; parameter-free).
2. Per group of 16 (within each output row): `group_scale = absmax`,
   `normalized = group / group_scale`.
3. Assign each normalized weight the nearest of 16 codebook levels. The
   codebook is the **frozen Lloyd-Max LUT for a unit Gaussian**
   (`_FIXED_LUT`, values ±0.9806…, ±0.7949…, …, ±0.0507) because ConvRot
   Gaussanizes rotated groups; a per-tensor k-means fit is only a fallback
   when excess kurtosis < -0.1.
4. Two ALS (alternating least-squares) passes refine `group_scale`:
   `scale = Σ(w·level) / Σ(level²)` then re-assign codes.
5. `s_channel[n] = absmax(reconstructed row) / 127`;
   `s_rel = group_scale / s_channel` cast to fp8-e4m3.
6. Codes are re-assigned against the **int8 grid**: the decoded value of code
   c in group g is `round(clamp(codebook[c] * s_rel[g], -127, 127))`, i.e.
   the weight is effectively snapped to the same int8 grid the GEMM consumes.

`[I]` The stored `weight_codebook` tensors match `_FIXED_LUT` (verified in
`artifacts/v2/metrics/w4a8_weight_parity.csv` analysis).

## 4. Dequantization (exact reference used for parity)

`[S]` `dequantize_w4a8_int8_weight`:

```
codes    = unpack(qdata)                        # [N, K] uint
w_int8   = round(clamp(codebook[codes] * s_rel.float(), -127, 127))   # [N, K] int8
w_rot    = w_int8.float() * s_channel[n]        # rotated-domain weight
W_hat    = w_rot @ H^T                          # un-rotate to original basis
```

## 5. Linear computation at inference

`[S]` `w4a8_int8_linear` (eager reference; no `correction` tensor exists in
this artifact):

1. Decode packed int4 → int8 grid on the fly (`_dequant_int4_grouped_to_int8`).
2. Reuse `int8_linear(x, w_int8, s_channel, convrot=True, groupsize=256)`:
   rotate activation `x @ H`, dynamic per-row int8, int32 GEMM, rescale by
   `x_scale * s_channel`.

So "A8" = the same dynamic per-row INT8 activation quantization as the INT8
ConvRot arm; "W4" = 16 non-uniform (Lloyd-Max) levels per 16-element group,
expressed on the int8 grid via an fp8 group scale and an fp32 channel scale.

## 6. Why "Asym" in the layout name

`[S]` The layout class (`AsymW4A8Int8Layout`) also supports asymmetric
uniform int4 (zero-point + `correction` tensor) — that path is NOT used by
this artifact: no `weight_correction`/zero-point tensors exist `[M]`, and the
codebook path is symmetric.

## 7. Verification hooks in this repo

- `artifacts/v2/manifests/tensor_index_c_w4a8.json` — full tensor index `[M]`
- `scripts/v2_weight_parity.py` → `artifacts/v2/metrics/w4a8_weight_parity.csv`
- Codebook provenance check → `artifacts/v2/metrics/w4a8_codebook_provenance.json`
