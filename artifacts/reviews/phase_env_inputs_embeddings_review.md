# Independent Verification Review — Environment / Inputs / Semantics / Embeddings / Layers

Reviewer: independent verification pass (no trust in main-agent claims; everything recomputed from raw artifacts or rerun).
Date: 2026-10-03. Scope: items 1–8 of the review brief.

## VERDICT: PASS WITH WARNINGS

All substantive numerical and semantic claims reproduce from raw artifacts. Two bookkeeping flaws
(overwritten metrics/meta files) reduce provenance quality but do not affect correctness of the frozen tensors.

---

## 1. Environment — VERIFIED

- `readlink -f /opt/rocm` re-executed by reviewer → `/opt/rocm-7.14.0` (matches `rocm_symlink.txt`).
- Reviewer-imported torch: `2.13.0+rocm7.14.0`, `torch.version.hip = 7.14.60850` (matches `summary.json`, `torch_check.txt`).
- `rocminfo.txt`: `gfx1100`, Marketing Name `AMD Radeon Pro W7900D`, ROCk module 6.16.13, 48 GiB VRAM. Consistent with `rocm-smi` artifacts and meta files (`device: AMD Radeon Pro W7900D`).
- HIP header (`rocm_hip_version_header.txt`) says 7.14.60850 — consistent.

## 2. Input identity — VERIFIED (independently recomputed)

- Reviewer re-tokenized all smoke prompts with the ComfyUI tokenizer
  (`/workspace/qwen3vl-cpu-benchmark/ComfyUI/comfy/text_encoders/qwen25_tokenizer`, `add_special_tokens=False`, T2I template)
  and compared to `artifacts/inputs/S0*.npz` `input_ids`:
  S01 (30 tok), S02 (40), S03 (36) — **EXACT** in all three; `drop_idx=14` = second `<|im_start|>` position in each.
- `input_alignment.csv` rows match my recomputation. `attention_mask` all ones; `mm_token_type_ids` all zeros (text-only) — confirmed directly in npz.
- Canary inputs C01–C10 also carry `drop_idx=14`.

## 3. Conditioning semantics — VERIFIED (tests RERUN on GPU by reviewer)

`tests/test_conditioning_semantics.py` rerun in full (`python -m tests.test_conditioning_semantics`): **ALL 3 TESTS PASS**.
- `test_hidden_states_semantics`: with the norm hook returning `args[0]`, `hidden_states[-1]` == captured prenorm **bitwise** (`torch.equal`), `hidden_states[36]` == prenorm **bitwise** (37 hidden states), postnorm differs from prenorm, drop_idx == second im_start == 14. This proves the semantics claim on the live stack (transformers 5.18).
- `test_pipeline_parity`: frozen `R0/S01.npz` `prompt_embeds` (cast fp32→bf16→fp32) **`torch.equal`** to a fresh `QwenImage21Pipeline.encode_prompt` output on CUDA — bit-match confirmed, not just allclose.
- `test_template_matches_pipeline`: template string present in installed pipeline source (weak-but-valid check).
- Prior log evidence of these tests running was absent — this is why I reran them. Now verified first-hand.

Code cross-check vs official `pipeline_qwenimage21.py::_get_qwen_prompt_embeds` (diffusers 0.40.0, lines ~234–330):
`run_rocm_encoder.py::encode_prompt` uses the identical hook (`text_model.norm` forward hook returning input), passes
`input_ids/attention_mask/mm_token_type_ids` from the processor-frozen npz, and derives `prompt_embeds = prenorm[drop_idx:]`.
Pipeline's mask-extract → drop is equivalent for batch=1 all-ones mask (verified: attn all ones). No nonstandard preprocessing found.

## 4. Embeddings — VERIFIED by recomputation (with a bookkeeping WARNING)

Reviewer recomputed cosine / relative-L2 from raw npz (`artifacts/tensors/*/S0*.npz`), float64:

`prompt_embeds` (kept tokens), S01 / S02 / S03:
| pair | cos (S02) | rel_l2 (S02) | cos range over S01/S03 |
|---|---|---|---|
| R0:R0R | 1.00000000 | 0.0 (bit-exact) | 1.0 / 1.0 — all bit-exact |
| R0:R1 | 0.99687550 | 0.07899764 | 0.99670318 / 0.99596960 |
| R1:O0 | 1.00000000 (8e-6) | 0.00000804 | ~1.0 / ~1.0 |
| R0:O0 | 0.99687540 | 0.07899894 | 0.99670306 / 0.99596946 |
| O0:O3 | 0.99781610 | 0.06606531 | 0.99689471 / 0.99730309 |
| R0:O3 | 0.99325019 | 0.11599581 | 0.99170052 / 0.99148519 |

`prenorm` (full seq) S02: R0:R0R = 1.0/0.0; R0:R1 = 0.99871108/0.05291506; R1:O0 = 1.0/7.6e-6;
R0:O0 = 0.99871101/0.05291723; O0:O3 = 0.99776283/0.06752380; R0:O3 = 0.99580549/0.09448992.

Internal consistency (all arms, S02): `|hs[36]−prenorm|max = 0` and `|prompt_embeds − prenorm[14:]|max = 0` — exact.

Saved-artifact fidelity: the only pair present in `embedding_summary.csv` / `embedding_summary_stats.json` (O0_vs_O1)
matches my recomputation to 16 significant digits (S02 pe cos 0.9999076358506833, rel_l2 0.0135944181827211).

**WARNING:** `embedding_summary.csv`/`embedding_summary_stats.json` were OVERWRITTEN (15:16) by a single-pair
(O0:O1) run of `compare_embeddings.py`; the six-pair evidence the script's defaults produce exists nowhere in
`artifacts/metrics/`. The numbers above are my independent recomputation from raw tensors (they corroborate the
surviving `layer_alignment_*` files, see §5).

Raw-dtype integrity: R0 `*_rawdtype.pt` prenorm/prompt_embeds are bfloat16; `raw.float() == npz fp32` bitwise;
`prompt_embeds_bf16_view` round-trips exactly through bf16.

## 5. Layer mapping — VERIFIED (code + numeric)

- `run_openvino_encoder.py::layer_output_ops` selects ops named `__module.model.language_model.layers.N/aten::add/Add_1`
  (N=0..35) — the final residual `Add` of each decoder layer; `find_prenorm_anchor` anchors prenorm at layer 35's `Add_1`.
  For O3/O1F (ComfyUI artifacts) outputs become `[logits, layer0..35, postnorm]`; `hs_list=[embeds]+outs[1:37]`,
  `prenorm=outs[36]`, `postnorm=outs[37]` — i.e. entry i = transformer layer i−1 output, entry 36 = layer-35 = prenorm.
- Numeric spot-check O3/S02: `hidden_states[36]` vs stored `prenorm` — **bitwise equal**. Same for all 7 arms.
- Cross-arm proof of mapping: R1 vs O0 per-layer (reviewer-computed from npz, S02): cos = 1.0000000 at layers
  0,1,10,20,30,36 with rel_l2 0 / 1.9e-6 / 3.3e-6 / 3.6e-6 / 3.6e-6 / 7.6e-6. **Off-by-one control** (O0[36] vs R1[35]) gives
  cos 0.778 — the mapping is unambiguous. Saved `layer_alignment_R1_vs_O0_summary.json` (layer36 rel_l2_median 7.9e-6) matches.
- O0 wrapper (`export_ov_clean.py`): same norm-hook neutralization, 39 outputs, `compress_to_fp16=False`, fp32 load —
  and `O0.hs[0]` is bitwise the token embeddings.
- Embeddings bridge: `R1.hs[0]` (PyTorch path) is **bit-exact** equal to `O0.token_embeddings` (numpy fp32 gather) —
  simultaneously proves identical input_ids fed to both arms and bit-exactness of the gather replacement.
  (O3 uses the artifact's own compressed embedding table: rel_l2 0.030 vs R1 — expected for the INT8 artifact, noted.)

## 6. Identical inputs across arms — VERIFIED

Both `run_rocm_encoder.py` and `run_openvino_encoder.py` read the identical frozen files
`artifacts/inputs/{pid}.npz` (same code path, reviewer-read source). Corroborating evidence:
R1↔O0 token embeddings bitwise identical (impossible unless input_ids identical); `drop_idx=14` stored in every arm's npz;
identical seq lengths; R0 vs R0R canary `prenorm_hash` equal for all 10 prompts (10/10 identical, 0 differ);
R0R smoke repeats S01/S01_rep1/S01_rep2 prenorm bitwise identical (GPU determinism noise floor = 0).

## 7. Preprocessing vs official pipeline — NO FLAWS FOUND

Line-by-line comparison against `_get_qwen_prompt_embeds`: template string identical (and asserted against installed
pipeline source), left padding only matters for batch>1 (batch=1 everywhere), `mm_token_type_ids` forwarded exactly when
present (all zeros text-only), norm hook identical, drop semantics identical, bf16-view cast matches pipeline dtype usage.
OV `build_feed` feeds `attention_mask=ones` (valid: masks verified all-ones) and `position_ids = arange` replicated on all
3 M-RoPE axes — correct for text-only Qwen3-VL; confirmed numerically by R1:O0 ≈ 1e-6 agreement from layer 1 upward.

## 8. Plots — VERIFIED

14 PNGs in `artifacts/plots/`, all valid PNGs (header-parsed), 980x700 / 1120x392 / 1120x700 px, 28–57 KB each — non-trivial content.

---

## Discrepancies found

1. **Metrics overwrite (main issue):** `embedding_summary.csv` + `embedding_summary_stats.json` contain ONLY `O0_vs_O1`;
   the multi-pair results were destroyed by a later single-pair run. Claimed pair numbers must be (and were) recomputed from tensors.
2. **Meta overwrite:** `meta_R0/R0R/R1.json` were overwritten by the 17:09 canary run — the smoke-run timing/hash metadata
   is lost (smoke npz tensors themselves survive, 13:40–13:50). Same pattern: outputs not versioned per run.
3. **Config vs run mismatch (numerics-neutral):** `experiment.yaml` `accuracy_props` specify `PERFORMANCE_HINT: "ACCURACY"`,
   but `run_openvino_encoder.py` only sets `EXECUTION_MODE_HINT=ACCURACY` and never `PERFORMANCE_HINT` (meta reports the
   LATENCY default). PERFORMANCE_HINT is scheduling-only; numerics demonstrably unaffected (R1:O0 ~8e-6, INFERENCE_PRECISION verified f32 for O0 / bf16 for O1).
4. **Environment symlink correction mid-capture:** pre-correction artifacts show `/opt/rocm → /opt/rocm-7.2.1` at 13:09;
   corrected to 7.14.0 by 13:25. All tensor runs postdate the correction; torch was already rocm7.14-built at 13:09.
5. Minor code cosmetics: dead unreachable code in `run_openvino_encoder.py` (lines 253–255); `sha_art` looks for
   `openvino_language_model.xml` so O0's meta `model_files` hash is null. No effect on results.
6. `test_template_matches_pipeline` is a source-string grep — weak, but the strong bit-match test (`test_pipeline_parity`) covers the substance.

## Risks / limitations

- Smoke suite is 3 prompts (canary 10); bf16-exact rates (~14% for O0:O1 prenorm) are sequence-length dependent — fine for alignment conclusions, not a production QA set.
- R0R determinism floor measured as exactly 0 (bit-identical repeats) — means GPU nondeterminism was never sampled across process restarts/flash state; the "noise floor" is same-process repeat noise only.
- The bf16 precision delta (R0:R1 rel_l2 ≈ 0.08 on prompt_embeds) dominates R0:O3 comparison; isolating INT8 error requires the O0:O3 column (0.066–0.079), which the artifacts do support.
- No archived log of the semantics test run prior to this review; I reran it and it passes — future runs should log test output to artifacts/logs.
- O0's token-embedding gather is numpy-side, not inside the OV graph (justified: bit-exact vs R1; documented in code), and O3's embedding table is the customer artifact's own compressed one — arm-to-arm embedding provenance differs by design and is disclosed in meta/npz (`token_embeddings`).

## Bottom line

Environment, input identity, conditioning semantics (rerun, bit-match), layer mapping (incl. off-by-one control),
cross-arm input identity, and all recomputed embedding metrics are correct and self-consistent. The two overwriting
incidents (metrics CSV/JSON, ROCm-arm meta) are the only real defects — provenance/bookkeeping, not numerics.
