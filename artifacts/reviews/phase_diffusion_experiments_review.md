# Independent Verification Review — Diffusion Experiments (Phases E/F/G/H)

Reviewer: independent verification pass. No trust in main-agent claims; everything re-derived from raw
artifacts (safetensors/npz/JSON/logs/PNGs), installed diffusers source, and reviewer-run recomputation (CPU).
Date: 2026-10-03. Scope: items 1–8 of the review brief. No project files were modified except this review.

## VERDICT: PASS WITH WARNINGS

All numerical results are genuine, internally consistent, and reproduce from raw tensors — including bitwise
cross-run reproductions I verified independently. Two flaws temper the pass: the designated self-determinism
test is **vacuous by construction** (its conclusion happens to be true, proven by other artifacts), and the
crossover conclusion **as phrased in the brief is inverted** relative to what the data shows. The underlying
crossover/perturbation data themselves are sound.

---

## 1. SELF-DETERMINISM — conclusion TRUE, designated evidence VACUOUS (warning)

- `artifacts/logs/diff_sanity.log`: two R0 runs (31.5 s, 25.5 s), both `timesteps_match_manifest=True`,
  then `final vs R0: latent cos 1.000000 rel_l2 0.000000`.
- **Flaw found in `scripts/run_diffusion_trace.py:101-121`**: `results` is keyed by arm name. With
  `--arm-a R0 --arm-b R0` the second run **overwrites** the first (`results["R0"] = ...`), so
  `step_stats` compares the second run's trace **with itself**. Proof from the artifact itself:
  `trace_S01_25s_selfcheck.json` has `arms: ["R0","R0"]` but exactly ONE entry each in `gen_s`, `n_steps`,
  `pe_hashes`; every step has `rmse == 0.0` and `max_abs == 0.0` exactly (cosine wobbles 1±2e-16 — fp64
  dot-product rounding of identical arrays). The logged 1.000000/0.000000 therefore proves nothing about
  run-to-run determinism.
- **Independent verification that the conclusion still holds (reviewer's own recomputation):**
  - MD5-identical PNGs from different processes ~2.5 h apart:
    `S02_25s_R0.png` ≡ `pert_S02_1024_25s_E0.png` (`0d8cf7b5…`), `S02_25s_O3.png` ≡ `pert_..._25s_E1xOVdelta.png`
    (`e6424ae4…`), and the same two 40-step pairs (`1d9a58f7…`, `36ae32f3…`). Same latent + embeds + steps →
    bitwise-identical images across processes.
  - `perturbation_S02.json` E1x finals equal `trace_S02_25s/40s.json` finals to **all 16 digits**
    (cos 0.9971271761056135 / rel_l2 0.0759396508013226; cos 0.9828737619979382 / rel_l2 0.18490493044688602).
  - Step-0 `model_out` metrics identical to 16 digits across the 25 s and 40 s processes
    (0.9998716861408655 / 0.01659318495558377).
- Net: bitwise self-determinism on this stack is real, but is established by cross-artifact evidence, not by
  the selfcheck run. Note `meta_R0.json` says `deterministic_algorithms: False` — determinism is empirical
  (kernel-level), not enforced; a ROCm/driver change could break it.

## 2. TRANSPLANT Phase E/F — VERIFIED

- Controls re-hashed by reviewer: sha256 of `initial_latent_1024.safetensors` latent (packed `[1, 4096, 64]`)
  = `976e549d…` — matches `controls.latent_hash` in `trace_S02_25s.json`, `trace_S02_40s.json`, and the
  selfcheck. `pe_hashes` for R0/S02 (`84c3e033…`) and O3/S02 (`2ab8fc13…`) recompute exactly from
  `artifacts/tensors/{R0,O3}/S02.npz` (both `prompt_embeds_bf16_view` = [26, 4096]); same hashes in both the
  25 s and 40 s JSONs. Timesteps recorded in both JSONs are **exactly** (list-equal) the manifest lists.
  `true_cfg: 1.0`, `negative: null`, `n_steps` 25/40 per arm. Only variable between arms = the embedding.
- Final divergence growth confirmed: 25 s final cos 0.997127 / rel_l2 **0.075940** → 40 s final cos 0.982874 /
  rel_l2 **0.184905**. (The brief's "0.0758 / 0.1847" actually quote the penultimate step-23/38 values printed
  in the log tail — step 24/39 are the finals. Cosmetic, see D4.)

## 3. CROSSOVER Phase G — data VERIFIED; stated conclusion INVERTED (warning)

Artifact `crossover_S02_k25.json` (matches `crossover_S02.log`, four continuations 25→40):
- Late-conditioning effect: `A_ref|B_swap_cond` rel_l2 **0.00333**, `C_swap_latent|D_ov` rel_l2 **0.00325**.
- Latent-state effect: `A_ref|C` 0.18518, `A_ref|D` **0.18516**, `B|C` 0.18521, `B|D` 0.18519 — all ≈ 0.185,
  i.e. the full transplant 40-step divergence (0.18490).
- Image-level PSNR (reviewer-computed): C|D and A|B ≈ 55 dB (visually identical); A|D = 17.25 dB — the same
  17.25 dB as pipeline R0|O3 at 40 steps.

**Interpretation check**: swapping conditioning in steps 25→40 changes the final by ~0.003 (washed out);
swapping the latent state at step 25 reproduces the entire 0.185 divergence. The data therefore shows
**divergence is carried by the accumulated latent state at step 25 (locked in by conditioning during steps
0–25); late-step ongoing conditioning contributes ~2% and is washed out.** The brief's phrasing — "divergence
driven by ongoing conditioning in late steps, accumulated state washed out" — states the **opposite** of what
the artifact shows (if accumulated state were washed out, C would converge to A; it does not: A|C = 0.185).
If the main report uses that phrasing, it must be corrected; the numbers themselves are correct.

Script audit `scripts/crossover_25_40.py` vs installed `pipeline_qwenimage21.py` (diffusers 0.37.0.dev0) and
`transformer_qwenimage21.py` — all verified:
- `make_schedule`: `sigmas = linspace(1.0, 1/steps, steps)`; `calculate_shift(seq_len=4096, …)` with
  `.get()` fallbacks (1.15/4096 never fire — snapshot config is base_shift 0.5, max_shift 0.9, base_seq 256,
  max_seq 8192 → mu 0.693548…); `retrieve_timesteps(s, steps, "cuda", sigmas, mu)` — identical to pipeline
  lines 723–733. Reviewer re-derived both 25/40 schedules on CPU: timesteps, sigmas AND mu match the frozen
  manifest **exactly**.
- `img_mask = cat(zeros(1, text_len=26, bool), ones(1, seq/4=1024, bool))` — exactly the pipeline's
  `image_pad_mask` for text-only prompt_embeds (`prompt_embeds.new_zeros(shape[:2])`, encode_prompt line 370)
  followed by `append_target_slots` (line 741: `mask.new_ones(batch, latents.shape[1]//4)`). Length 26+1024;
  transformer expands each image slot 4× → 26+4096 joint tokens.
- `img_shapes=[[ (1, 64, 64) ]]` matches pipeline t2i layout (lines 712–720); `timestep/1000`; transformer
  called with `return_dict=False`, `encoder_hidden_states_mask=None`; `noise_pred[:, -lat.size(1):]` slicing
  identical to pipeline line 784 (output is full joint seq from `proj_out`).
- Scheduler continuation: `sched.set_begin_index(0)` phase 1, `sched.set_begin_index(k)` phase 2 with
  `run_steps(k, 40)`. Verified in installed `FlowMatchEulerDiscreteScheduler`: `_init_step_index` uses
  `self._begin_index` when set → first continuation step consumes `sigmas[25], sigmas[26]`, exactly the
  pipeline's continuation; Euler step `prev = sample + dt*model_output` in fp32 then cast back.
- `torch.no_grad()` wraps both phases; per-branch fresh scheduler + bitwise `lat0` from safetensors; embeds
  re-loaded per branch. VAE decode replicates `_unpack_latents` (transpose/reshape), `*latents_std +
  latents_mean`, `decode(...)[:, :, 0]`, `postprocess` — faithful.
- **Caveat (D3)**: crossover calls the transformer **without kv_cache**, while the pipeline traces ran with
  the default `use_kv_cache=True`. The official docstring explicitly warns the two settings give "visibly
  distinct samples" in reduced precision — reviewer measured crossover `A_ref` vs pipeline `S02_40s_R0.png`:
  **31.90 dB** PSNR (D_ov vs S02_40s_O3: 33.60 dB). The four branches are internally consistent (all no-cache),
  and the regime transfer is corroborated by A|D (0.18516) ≈ transplant final (0.18490, Δ=2.6e-4), but the
  crossover strictly characterizes the no-cache regime.

## 4. PERTURBATION Phase H — VERIFIED

- Recomputed from raw npz: `delta_norm` 482.7848815917969 (exact match), delta rel_l2 0.115956 / cos 0.993255
  (match). **E1xOVdelta = E0 + 1.0·(E1−E0) equals O3 embeds 100.0000% bitwise, both in fp32 and after the bf16
  cast** — so the E1x arm IS the O3 transplant arm, and its finals being bitwise-equal to the transplant
  finals additionally proves cross-process determinism.
- Norm-matching code verified and reproduced for seeds 11/22/33: per-token L2 norms of the random delta equal
  delta's within fp32 rounding (max 3.05e-5 on ~482-norm vectors); global Frobenius norm 482.786 ≈ 482.785.
  The perturbation is genuinely magnitude-matched, per-token.
- E0 baseline: same frozen latent file and same step counts; proven bitwise — `pert_..._E0.png` md5-identical
  to `S02_{25,40}s_R0.png`.
- Numbers in the brief match the JSON exactly: E1x 40 s rel_l2 0.18490 / PSNR 18.50; R11/R22/R33 40 s:
  0.23496/17.97, 0.16734/19.62, 0.17717/19.40. E1x sits inside the random band at 40 steps → "OV delta
  behaves like a generic norm-matched perturbation" is supported there. Minor caveat (D5): at **25** steps
  E1x (0.0759) is *below* all three randoms (0.117/0.117/0.183) with only n=3 seeds — milder than generic;
  the claim is cleanest in the 40-step case the brief cites.

## 5. PIPELINE-SEMANTICS PARITY of run_diffusion_trace.py — VERIFIED

Line-by-line vs `QwenImage21Pipeline.__call__`:
- Latent: loaded from safetensors (packed `[1, 4096, 64]`, fp32) → `.to("cuda", bfloat16).clone()` per arm;
  `prepare_latents` (line 483) passes provided latents through with only device/dtype cast → injected
  bitwise-identically for every arm/run. Deterministic cast ⇒ "frozen latent bitwise" holds.
- `prompt=None` + `prompt_embeds` (bf16) is the sanctioned path (`check_inputs` lines 404–407);
  `prompt_embeds_mask=None` → text-only zeros `image_pad_mask` (encode_prompt line 370), batch 1, no padding.
- **No generator**: none passed; `randn_tensor` unreachable (latents provided) and `stochastic_sampling=False`
  in the scheduler config ⇒ no RNG anywhere in the path.
- `true_cfg_scale=1.0` + `negative_prompt=None` → `do_true_cfg=False` (line 669): single forward per step.
- Callback: `callback_on_step_end` with `["latents"]` (validated against `_callback_tensor_inputs`);
  captures post-scheduler-step latents; count asserted equal to model_out count (25/40).
- Forward hook on `pipe.transformer` (observer only, returns output unchanged) slices
  `output[0][:, -lat.size(1):]` — identical to pipeline line 784.
- Per-arm scheduler reset (`from_config`) ⇒ identical timesteps per arm; `height=width=1024` (multiple of 32).
- Trace runs use the pipeline's default `use_kv_cache=True`, i.e. the official sampling path unmodified.

## 6. TRANSPLANT LOG CONTROLS + MANIFEST — VERIFIED

- `transplant_S02_25.log`: `[R0] … timesteps_match_manifest=True` and `[O3] … timesteps_match_manifest=True`;
  `transplant_S02_40.log` likewise (4/4 lines present).
- Manifest 1024: 25-step vs 40-step schedules genuinely differ — reviewer recomputation:
  `max|t25 − t40[:25]| = 538.55` (timestep units), `max|σ25[:25] − σ40[:25]| = 0.539`;
  `schedule_overlap_check.json` flags both overlap tests False. "25 steps" is a distinct schedule, not a
  truncation of 40 (both correctly end at the shared shift_terminal timestep 19.99998).
- Both schedules re-derived from scratch with the installed diffusers (snapshot scheduler config, linspace
  sigmas, `calculate_shift`, `retrieve_timesteps`, CPU): timesteps, sigmas, and mu (0.6935483870967742)
  match the manifest exactly.

## 7. STEP-STAT MONOTONICITY — VERIFIED (both traces, every step)

- `trace_S02_25s.json` (25 steps) and `trace_S02_40s.json` (40 steps): `latent.rel_l2` **strictly increasing**
  and `latent.cosine` **strictly decreasing** at every step (25 s: 0.00151→0.07594; 40 s: 0.00116→0.18490).
- Geometric feasibility `cos ≥ sqrt(1−rel_l2²)` holds at every step (both traces); implied ‖b‖/‖a‖ ∈
  [0.968, 1.000]; implied RMS(reference latent) trajectory smooth (0.78→1.24) — all self-consistent.

## 8. IMAGES — ALL PRESENT

`S02_25s_R0/O3.png`, `S02_40s_R0/O3.png`, `crossover_S02_40s_k25_{A_ref,B_swap_cond,C_swap_latent,D_ov}.png`,
and all 16 `pert_S02_1024_{25s,40s}_*.png` exist, valid 1024×1024 PNGs. Cross-run md5 identities listed in §1/§4.
R0 vs O3 differ (25 s: 25.31 dB; 40 s: 17.25 dB image PSNR — reviewer-computed), consistent with the latent metrics.

---

## Discrepancies / risks

- **D1 (evidence flaw)**: Self-determinism selfcheck is vacuous — `results` dict keyed by arm name collapses
  the R0-vs-R0 pair into a self-comparison (`run_diffusion_trace.py` results/step_stats loop; single-entry
  `gen_s`/`pe_hashes` in the selfcheck JSON prove it). Conclusion independently confirmed (§1), but the
  designated artifact must not be cited as proof. A fix needs distinct keys or a dedicated two-run comparer.
- **D2 (interpretation error)**: The crossover conclusion as phrased ("divergence driven by ongoing
  conditioning in late steps, accumulated state washed out") is the inverse of the measured behavior; correct
  reading: accumulated latent state at step 25 carries the full 0.185; late-step conditioning is washed out
  (~0.003). The JSON itself needs no correction — only the narrative.
- **D3 (regime mismatch)**: Crossover/perturbation-branch machinery runs the transformer without KV cache;
  pipeline traces run with cache (default). Measured effect of the toggle ≈ 32–34 dB image PSNR. Internal
  comparisons unaffected; quantitative transfer to the cached regime rests on the observed A|D ≈ transplant
  final agreement (Δ 2.6e-4).
- **D4 (quoting slip)**: Brief's transplant rel_l2 values (0.0758 / 0.1847) are the penultimate-step log
  prints; true finals are 0.075940 / 0.184905. Substance unaffected.
- **D5 (minor)**: Perturbation random baseline uses n=3 seeds with wide spread (0.167–0.235); at 25 steps the
  OV delta falls below the sampled random band. "Generic perturbation" equivalence is solid at 40 steps only.
- **R1**: Determinism is empirical (no `torch.use_deterministic_algorithms`), single GPU/driver stack;
  reproducibility could break on ROCm/kernel updates.
- **R2**: All diffusion evidence is prompt S02 at 1024² (plus S01 selfcheck); the brief's claims are scoped
  accordingly — no broader suite was claimed, so this is a scope note, not a defect.

## Bottom line

Every load-bearing number in the four experiment groups reproduces from raw tensors — several to bitwise
exactness across independent processes — and the trace/crossover/perturbation scripts faithfully implement
the official QwenImage21 sampling semantics (schedule, mask layout, slicing, scheduler continuation, no-RNG,
cfg=1). Verdict PASS WITH WARNINGS due to D1 (vacuous designated self-determinism evidence) and D2 (inverted
crossover conclusion as worded in the brief), plus the D3 cache-regime caveat.
