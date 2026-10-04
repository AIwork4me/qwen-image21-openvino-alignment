# Independent Verification Review — Follow-up Steps 3 & 4

Reviewer role: final independent verification (raw-artifact recomputation; no project files modified)
Date: 2026-10-04
Scope: Step 3 (2K BF16 spot check) and Step 4 (customer artifact IR root cause + runtime patch O3p) of the follow-up plan, per CONCLUSIONS.md §7–8, reports/CUSTOMER_REPORT.md §16, reports/TECHNICAL_REPORT.md §15.

## VERDICT: PASS WITH WARNINGS

Every quantitative claim recomputed from raw artifacts matches (several to the last printed digit). The
bitwise-equivalence claim of Step 4 was independently reproduced. Warnings are documentation/provenance
level only (listed in §Discrepancies); none affects the engineering conclusions.

Environment used for verification: OpenVINO 2026.4.1-22982, torch 2.13.0+rocm7.14.0 (CPU tensor path),
lpips 0.1.4 (net='alex'), numpy 2.1.2 — same stack as the project environment snapshot.

---

## Step 3 — 2K BF16 spot check

### 3.1 Artifacts exist
- Images (2048×2048, verified by pixel-array shape on load): `production_30_P{01,03,12,21}_s20261001_40s_{R0,O1}.png`
  (4 prompts × 2 arms) plus `..._O3r.png` for all 30 production prompts. O1 PNGs dated Oct 4 13:05;
  R0/O3r dated Oct 3 — consistent with the follow-up timeline.
- O1 embedding tensors: `artifacts/tensors/O1/P01..P30.npz` all present (30/30), full schema
  (hidden_states/prenorm/postnorm/prompt_embeds/prompt_embeds_bf16_view/drop_idx/token_embeddings).
- Frozen 2K latent: `artifacts/latents/initial_latent_2048.safetensors` + seed/schedule manifest
  (seed 20261001). Usage-not-rerun: reliance on the frozen-latent mechanism previously verified (step2 review).

### 3.2 LPIPS recomputed from PNGs (lpips alex, a/127.5−1, full-res, CPU)
Independently reimplemented per scripts/image_metrics.py:33-37 preprocessing; all values identical to
`image_metrics_production_30_P__s20261001_40s_R0_vs_{O1,O3r}.csv` to every printed digit:

| Pair | CSV | My recomputation |
|---|---|---|
| P01 R0↔O1 | 0.01717647723853588 | 0.01717647723853588 |
| P03 R0↔O1 | 0.044511839747428894 | 0.044511839747428894 |
| P12 R0↔O1 | 0.009327943436801434 | 0.009327943436801434 |
| P21 R0↔O1 | 0.13186170160770416 | 0.13186170160770416 |
| P01 R0↔O3r | 0.12131720036268234 | 0.12131720036268234 |
| P03 R0↔O3r | 0.05606452748179436 | 0.05606452748179436 |
| P12 R0↔O3r | 0.03390186280012131 | 0.03390186280012131 |
| P21 R0↔O3r | 0.21971137821674347 | 0.21971137821674347 |

- R0↔O1 (n=4): median 0.030844, max 0.13186 → claim "median 0.031 / max 0.132" ✓
- R0↔O3r: max 0.21971 (P21, in subset) → claim "max 0.220" ✓. NOTE: claimed median 0.032 is the
  full n=30 2K-suite median (0.032437 in the summary json), NOT the 4-prompt subset (subset median
  0.0887). See Discrepancies W1.

### 3.3 O1 = fp32 weights + runtime bf16
`artifacts/tensors/O1/meta_O1_production_30.json`: artifact = `/valdata/models/ov_clean_fp32`
(`language_model_fp32.xml`, sha 43e0ecf8…), `runtime_precision_requested = "bf16"`,
`inference_precision = <Type: 'bfloat16'>`, 30/30 entries. Code path confirmed in
scripts/run_openvino_encoder.py:114,152-153 (INFERENCE_PRECISION_HINT=bf16 on the fp32 export). ✓

---

## Step 4 — customer artifact IR root cause + runtime patch (O3p)

### 4.1 Graph-structure root cause — CONFIRMED by direct IR inspection
Reading the unpatched `openvino_language_model.xml` via ov.Core (no files modified):
- `__module.model.language_model/aten::add/Add_3` (and Add_4/Add_5): input0 = `aten::index/GatherND`
  (fed by `Transpose` ← `NonZero` ← Parameter `visual_pos_masks`), input1 = `aten::select/Gather` on
  Parameter `deepstack_visual_embeds` with baked scalar Constant index (value 0) — exactly as claimed.
- Model inputs are dynamic (`[?,?,…]`) so the empty-tensor path hits shape inference at infer time.

### 4.2 Failure reproduced on the REAL unpatched artifact (this host, today)
Text-only feeds (mask all-False, deepstack [1,0,4096]) fail at exactly the claimed node:
`[CPU] Add node '__module.model.language_model/aten::add/Add_3' Check 'input_shape[j] == 1' failed at
src/plugins/intel_cpu/src/shape_inference/custom/eltwise.cpp:52` — 5/5 prompts tested
(A071=79 tok, A073=57, A072=54, A001=30, S01=30). ≥41-token deterministic failure: 3/3 ✓.
Short (30-token) prompts also fail today — consistent with the reports' own note that the unpatched
artifact no longer reproduces its original success on this host (environment drift); the historical
"shorter = nondeterministic" regime is no longer observable, which the reports acknowledge (see W2).

### 4.3 Runtime-only patch — artifact files verifiably untouched
- scripts/patch_ov_artifact_textonly.py contains no file I/O whatsoever (in-memory graph surgery only);
  the O3p path in run_openvino_encoder.py writes only to artifacts/tensors/O3p. My reproduction patched
  exactly 3 Adds (assert n≥3 satisfied), matching log records.
- sha256 of ALL 8 files in /workspace/qwen3vl-cpu-benchmark/models/qwen3vl-openvino-int8 recomputed and
  equal to artifacts/model_manifest.json `openvino_artifacts.customer_int8`, including both large bins:
  openvino_language_model.xml e09c1469… ✓, .bin 755ff48c… ✓ (7.57 GB),
  openvino_text_embeddings_model.xml a75283a7… ✓, .bin 391a745c… ✓, config/openvino_config/
  tokenizer_config/dl.log ✓. mtimes all 2026-09-29 (predate all Oct 3–4 follow-up work). ✓

### 4.4 Bitwise equivalence proof — INDEPENDENTLY REPRODUCED (exact match: YES)
Mirrored the O3p path (read model → apply_textonly_patch → find_prenorm_anchor/layer_output_ops →
add_outputs → compile with EXECUTION_MODE=ACCURACY / LATENCY / f32), embeddings from the artifact's own
`openvino_text_embeddings_model`, inputs from `artifacts/inputs/S0*.npz`:

| pid | shape | bitwise equal vs `artifacts/tensors/O3/S0*.npz['prenorm']` | max_abs | sha256(raw fp32 bytes) |
|---|---|---|---|---|
| S01 | (30,4096) | YES | 0.0 | f42891a9… = frozen |
| S02 | (40,4096) | YES | 0.0 | adc8f84e… = frozen |
| S03 | (36,4096) | YES | 0.0 | 0d5d672c… = frozen |

Provenance chain closes: common.tensor_hash of frozen O3 prenorm = 5313e9fd… = meta_O3.json entry =
meta_O3p_smoke.json entry (S02/S03 likewise). The frozen O3 tensors are the artifact's own historical
outputs; the patched model reproduces them bit-for-bit. ✓

### 4.5 n=100 coverage medians — recomputed from npz 'prenorm' (independent float64 implementation)
All 100 A-prompts present in O0/O3p/O3r/R0 (asserted); rel_l2 = ‖a−b‖₂/‖a‖₂:

| pair | claim | my recomputation | stats json |
|---|---|---|---|
| O0↔O3p rel_l2 median | 0.0689 | 0.068927 | 0.068927 ✓ |
| R0↔O3p rel_l2 median | 0.0965 | 0.096475 | 0.096475 ✓ |
| O3p↔O3r rel_l2 median | 0.0054 | 0.005387 | 0.005387 ✓ |
| O3p↔O3r cosine median | 0.99999 | 0.9999855 | 0.9999855 ✓ |

(means/mins/maxes also match: e.g. O0↔O3p mean 0.068572/max 0.071394; R0↔O3p max 0.098965;
O3p↔O3r min 0.004809/max 0.007164 — all equal to embedding_summary_stats_O3pO3r_O0O3p_R0O3p_alignment_100.json.)

### 4.6 O3p meta json
`meta_O3p_alignment_100.json` records the artifact path and `language_model` sha256 e09c1469… (matches
manifest and my recomputation) ✓ — but contains NO field recording that the text-only patch was applied
(checked all three O3p metas; the string "patch" is absent). The patch application is recorded only in
`artifacts/logs/o3p_enc.log` ("[O3p] applied runtime text-only patch to 3 visual-branch Adds", 3 runs:
smoke/canary/alignment_100). See W3.

---

## Report spot-checks (12 numbers, all consistent with recomputation)
1. CONCLUSIONS §7 Step 3: R0↔O1 2K LPIPS med 0.031 → 0.030844 ✓
2. CONCLUSIONS §7 Step 3: max 0.132 → 0.131862 ✓
3. CONCLUSIONS §7 Step 3 / TECH §15: R0↔O3r 0.032 → 0.032437 (n=30 suite; see W1) ✓
4. …max 0.220 → 0.219711 (P21) ✓
5. CONCLUSIONS §7 / TECH §15 Step 4: O0↔O3p 6.89% → 0.068927 ✓
6. R0↔O3p 9.65% → 0.096475 ✓
7. O3p↔O3r 0.54% → 0.005387 ✓; cos 0.99999 → 0.9999855 ✓
8. "bitwise identical, max_abs=0.0, S01–S03" → reproduced myself, 3/3 ✓
9. CUSTOMER_REPORT §16.3: "R0↔O3p = 9.65%, matching O3r within 0.54%" ✓
10. CUSTOMER_REPORT §16.2 worst-case 40-step LPIPS 0.098 (BF16) → canary R0↔O1 max 0.097615 ✓
11. CUSTOMER_REPORT §16.2 / CONCLUSIONS §7: BF16 canary median 0.0231 → 0.023086 ✓
12. TECH §15 Step 4: ">40 tokens deterministic" → 3/3 (54/57/79 tok) fail at Add_3 ✓; "n≥3 Adds patched" → exactly 3 ✓

## Discrepancies / warnings (none invalidates the conclusions)
- W1 (presentation): the Step-3 "R0↔O3r median 0.032" is the full 30-prompt 2K-suite median; on the
  4-prompt spot-check subset the median is 0.0887 (max 0.2197 is from P21, which is in the subset).
  Reports label these as context numbers only implicitly; a one-word "(n=30)" would remove ambiguity.
- W2 (unverifiable-historical, self-acknowledged): "shorter prompts fail nondeterministically" can no
  longer be probed — on this host today even 30-token prompts fail deterministically. The reports
  themselves state the unpatched artifact no longer reproduces its first success (environment drift),
  so the claim is consistent but not independently re-confirmable.
- W3 (provenance gap, minor): O3p meta jsons record the artifact sha but not the patch application
  (count/identity of patched nodes lives only in artifacts/logs/o3p_enc.log). Recommend adding a
  `textonly_patch: {applied: true, adds_patched: 3, script_sha: …}` field to future O3p metas.
- W4 (scope note): 2K "frozen latents" verified via the latent/schedule manifest and prior step-2
  verification of the frozen-latent mechanism, not by re-running the 320 s/per-image pipeline.

## Bottom line
Step 3 and Step 4 claims are fully supported by the raw artifacts: recomputed LPIPS values are
digit-identical, the IR root cause is confirmed by direct graph inspection and by reproducing the
exact OV CPU eltwise shape-inference failure on the untouched customer artifact, the patch is
provably runtime-only (all 8 artifact file hashes match the manifest), the patched model's prenorm
output is bitwise identical (max_abs = 0.0) to the artifact's own frozen historical outputs on
S01–S03, and the three n=100 coverage medians match my independent recomputation exactly.
