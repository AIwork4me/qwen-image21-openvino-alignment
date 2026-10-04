# Independent Verification Review — STEP 1 (accuracy-aware INT8 re-quantization: O3q / O3qL)

Reviewer method: everything below was recomputed from raw artifacts (saved IRs, npz tensors, manifests, runner source) with independently written scripts. No project files were modified except this review. Environment matches the experiment: openvino 2026.4.1-22982, nncf 3.4.0, numpy 2.1.2.

## VERDICT: PASS WITH WARNINGS

All four claimed findings reproduce from raw artifacts. The warnings are provenance nits (missing run-metas for the older O0/R0 smoke tensors; null model-file sha in int8 run metas) that do not affect any conclusion — details in "Discrepancies & observations".

---

## (a) Structural verification of the saved IRs (my own recount)

My script (full BFS over Constant consumers, no hop limit, scope = first consumer matching `layers\.(\d+)\.`):

| IR | total i8/u8 Constants | by dtype | layers.0–34 | layers.35 | other scope |
|---|---|---|---|---|---|
| `ov_clean_int8_acc_emb/language_model_int8.xml` (O3q)  | **504** | u8 ×504 | **490** | **14** | 0 |
| `ov_clean_int8_acc_full/language_model_int8.xml` (O3qL) | **490** | u8 ×490 | **490** | **0**  | 0 |

- Exactly 14 u8 Constants per transformer layer (7 linears × {weight, zero_point}) — perfectly uniform across all layers that are compressed; every Constant's scope-resolving consumer is 1 hop away. **Claim 1's counts (504 = 490+14; 490 = 490+0) confirmed exactly.**
- Bin sizes: emb = 6,951,203,821 B = 6.4738 GiB; full = 7,529,901,037 B = 7.0128 GiB — matches manifests and claimed "O3r 6.47 GiB / full 7.01 GiB" (O3r bin is also 6,951,203,821 B).
- **Notable (strengthens the work, worth stating explicitly):** the O3q `language_model_int8.xml` is **bit-identical (sha256 `8d164ca9…`) to the O3r IR**, and the two `.bin`s are the same size. This is expected and correct: the wrapper graph input is `inputs_embeds [1,1..2048,4096]` (embedding table is *not* in the language-model graph), so "embedding kept FP32" is realized by the runner's numpy gather (safetensors bf16→fp32), not by nncf. Consequently O3r vs O3q is a clean single-variable experiment: identical INT8 trunk, only the embedding path differs.
- Conversion manifests for both variants record source `/valdata/models/ov_clean_fp32`, `nncf 3.4.0 compress_weights`, INT8_ASYM ratio=1.0 group_size=-1; full additionally records ignored pattern `__module\.lm\.layers\.35\..*` — consistent with the export script (`scripts/export_ov_int8_acc.py:32-43`).

## (b) Smoke-suite embedding metrics — recomputed from npz (median over S01–S03, `prenorm` key; rel_l2 = ‖a−b‖₂/‖a‖₂)

| Pair (prenorm) | Claimed rel_l2 | My rel_l2 | Claimed cos | My cos |
|---|---|---|---|---|
| O0_vs_O3q  | 0.0707 | **0.07072** | 0.99756 | **0.997557** |
| O0_vs_O3qL | 0.0619 | **0.06191** | 0.99814 | **0.998144** |
| R0_vs_O3q  | 0.0984 | **0.09837** | — | 0.995708 |
| R0_vs_O3qL | 0.0934 | **0.09342** | — | 0.996163 |
| O3r_vs_O3q | 0.0037 | **0.00372** | — | 0.999993 |

- `prompt_embeds` key also recomputed (e.g. O0_vs_O3q median 0.07448, cos 0.997231) — matches the stored stats JSON.
- Baselines: alignment-100 files give O0_vs_O3r = 0.06961 (n=100) and R0_vs_O3r = 0.09667 (n=100) — matches claimed "0.0696 / 0.0967 (n=100)". (Smoke-suite O0_vs_O3r is 0.07046, n=3 — the claim correctly labeled the baselines as n=100.)
- All numbers in `embedding_summary_stats_O0O3q_…_smoke.json` reproduce from the npz files; per-prompt values match the per-token CSVs.

## (c) Layer-0 exactness and layer attribution

- `hidden_states[0]` (token embeddings) is **bitwise identical** O0 == O3q and O0 == O3qL (and O0 == R0) for all of S01/S02/S03; `prenorm` differs (rel_l2 ≈ 6–7%). **Confirmed.**
- O3r layer-0 rel_l2 (recomputed, smoke): 0.02980 / 0.02790 / 0.02664 → median **2.79%**; the stored `layer_alignment_O0_vs_O3r_summary.json` (larger suite) says **2.83%** — consistent with the claimed "≈2.8–3% at layer 0 in O3r / was 2.83%".
- Per-layer profile O0_vs_O3q recomputed (median of 3): L0 = 0.00000, L1 = 0.00698, layers 1–34 ∈ [0.0070, 0.0126] ("trunk 0.7–1.2%/layer" ✓), L35 = 0.02569 (visible jump, flagged in `significant_jumps`), final entry = **0.07072** ✓ (claimed 7.07%). Matches `layer_alignment_O0_vs_O3q_summary.json` row-for-row.
- Conclusion supported: with the embedding exact (layer-0 error 0.0), the 7% output delta must originate in the compressed trunk; excluding only layers.35 (O3qL = 6.19% vs O3q 7.07% / O3r 6.96–7.05%) recovers ~0.9pp of ~7pp, i.e. naive scope exclusion is indeed ineffective. The O3r_vs_O3q delta of 0.37% is exactly the marginal effect of the int8 embedding table propagated through an otherwise bit-identical INT8 trunk.

## (d) Input identity & runner code path

- All arms read the *same* frozen files `artifacts/inputs/S0{1,2,3}.npz` (single shared directory; runner `run_openvino_encoder.py:185`), seq lens 30/40/36, drop_idx=14 in every arm's npz.
- Runner source: O3q/O3qL take the `is_wrapper=True` branch (line 133–140) and the fp32 gather `embed_lookup_fp32(snap_path, input_ids)` **without** the int8 table (line 196–197); only O3r passes `int8_table` from `embed_table_int8.npz` (line 190–194). Code confirmed; and the bitwise `hidden_states[0]` equality O0==O3q==O3qL (vs 2.7–3.0% error for O3r) is runtime proof the exact-fp32 path was actually used.

## (e) NNCF constraint claim (tiny 2-layer MLP, torch→ov.convert_model, NOT the 8B model)

- `nncf.compress_weights(mode=INT8_ASYM, ratio=1.0, group_size=-1)` → works (2/2 layers int8_asym per-channel).
- `nncf.compress_weights(mode=INT8_ASYM, ratio=0.6, group_size=-1)` → **raises `nncf.errors.ParameterNotSupportedError`: "INT8 modes require per-channel quantization of all layers in 8 bit. Default values of `ratio` (1) and `group_size` (-1) cannot be overridden."** — claim confirmed verbatim on nncf 3.4.0.
- Control: `INT4_ASYM, ratio=0.6` is accepted → mixed-precision ratio genuinely is INT4-only. So "no accuracy knob within INT8 weight compression" is correct as stated (within data-free `compress_weights`; INT8-mode dropdowns like ignored_scope remain, which is exactly what O3qL tested).

## (f) Methodology / single-variable check

- Run metas for O3r / O3q / O3qL smoke: identical `exec_mode=ExecutionMode.ACCURACY`, `inference_precision=float32`, `runtime_precision_requested=f32`, same openvino build. (`hint: LATENCY` in the metas is just the read-back default of PERFORMANCE_HINT, which the runner never sets — same for all arms, so not a confound.)
- Same model source for all three arms (`/valdata/models/ov_clean_fp32/language_model_fp32.xml` per manifests); O3r and O3q IRs are bit-identical, so the O3r↔O3q comparison isolates the embedding variable exactly; O3q↔O3qL isolates the layers.35 exclusion (the only recipe difference, per manifests + my recount).
- No accidental variable changes found. The only cosmetic runner oddity (`if "_int8_tbl" not in dir()` lazy-cache, line 191–194) works as intended within a run and only affects the O3r arm's embedding table load.

## Discrepancies & observations (all minor)

1. **Provenance gap (warning):** `artifacts/tensors/O0/S0*.npz` and `R0/S0*.npz` have no corresponding `meta_O0_smoke.json` / `meta_R0_smoke.json` (O0's metas cover canary/alignment-100 only; the S-files date from Oct 3 15:04, earlier than the other arms' runs). Mitigated: shapes, drop_idx, bitwise token-embedding identity with the O3q/O3qL runs and the ROCm R0 arm confirm same inputs and same fp32 gather, and my recomputation used these files directly.
2. **Provenance gap (warning):** run metas for int8 arms record `model_files.language_model: null` — `sha_art()` in the runner looks for `openvino_language_model.xml`, which does not exist in the nncf export dirs. Mitigated by my own sha256 of the IRs and the conversion manifests.
3. Presentation nit: the claim's baseline row mixes n=100 numbers (O0_vs_O3r 0.0696) with n=3 smoke rows; smoke n=3 gives 0.07046. Values are consistent, but the n differs within one table (correctly labeled "(n=100)" in the claim).
4. Presentation nit: "embedding table kept FP32" for O3q is a property of the wrapper+runner design (table lives outside the graph; `inputs_embeds` fed from safetensors), not of nncf settings — the emb and O3r IRs are literally the same file content. The manifests word this correctly ("numpy gather").
5. Per-token CSV sanity: token-0 cosine identical across prompts (0.96329 for O0_vs_O3q on S01–S03) — expected, since token 0 is the shared BOS `<|im_start|>` at position 0; not an artifact-generation bug.

## Recomputed-numbers appendix

- int8 Constant recount: see table in (a).
- prenorm medians (rel_l2 | cos): O0:O3q 0.07072|0.997557; O0:O3qL 0.06191|0.998144; R0:O3q 0.09837|0.995708; R0:O3qL 0.09342|0.996163; O3r:O3q 0.00372|0.999993; (O0:O3r smoke 0.07046|0.997573).
- prompt_embeds medians (rel_l2): 0.07448 / 0.07143 / 0.13008 / 0.12938 / 0.00586.
- Layer-0: O0==O3q bitwise True (×3 prompts); O0_vs_O3r layer-0 rel_l2 = 2.98/2.79/2.66% (median 2.79%).
- Per-layer (median): L0 0.00000, L1 0.00698, L1–L34 max 0.01264, L35 0.02569, L36 0.07072.
- sha256: `ov_clean_int8/language_model_int8.xml` == `ov_clean_int8_acc_emb/language_model_int8.xml` = `8d164ca97ee9671eb738dcb5371a5b6f1e4c700f3f2210778d04b17a9cf1c40b`; bins 6,951,203,821 B (O3r, emb) and 7,529,901,037 B (full).
- NNCF tiny-model test: ParameterNotSupportedError reproduced for INT8 ratio=0.6; INT4 ratio=0.6 accepted.
