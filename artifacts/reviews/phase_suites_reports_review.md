# Independent Verification Review — Statistical Suites, Image Metrics, and Reports

Reviewer: final independent verification pass. No trust in prior claims or main-agent
narrative; everything re-derived from raw artifacts (PNGs / npz / safetensors / CSV / JSON),
with independent recomputation (hashing, PSNR/SSIM/LPIPS from pixels, Wilcoxon from CER
columns, embedding stats from tensors). No project files modified except this review.
Date: 2026-10-04.

## VERDICT: PASS WITH WARNINGS

Every load-bearing number I traced in the four reports reproduces from raw artifacts —
including full independent recomputation of embedding stats (3 prompts to 1e-12), 4 image
metric rows from pixels (PSNR/SSIM/LPIPS to 4 decimals), 16/16 manifest hashes, the OCR
worse/tie/better counts, the Wilcoxon/sign tests, and the crossover/perturbation tables.
The corrected crossover interpretation is stated consistently in all three narrative
reports. The warnings are bookkeeping/report-quality, not numerics: (1) the canary manifest
and the canary seed-20261001 40-step metrics CSV were truncated/overwritten by later runs
(same defect family the two prior reviews found); (2) the executive summary contains "n/a"
placeholders where measured numbers exist; (3) the technical report's CPU-performance table
is captioned "accuracy configs" while the artifact records EXECUTION_MODE_HINT=PERFORMANCE;
(4) the gallery lacks the crossover/perturbation categories.

---

## 1. SUITE COVERAGE & MANIFESTS — VERIFIED (with overwrite WARNING)

Image counts and dimensions (PIL-verified, all valid PNGs):
- **canary**: 120 files = 3 seeds (20261001/02/03) × 10 prompts × {25,40} × {R0,O3r}, all
  1024×1024. Seed split exactly 40/40/40.
- **alignment_100**: 400 files = 1 seed (20261001) × 100 prompts × {25,40} × {R0,O3r}, all
  1024×1024.
- **production_30**: 60 files = 1 seed × 30 prompts × 40 steps × {R0,O3r}, all **2048×2048**.

`suite_canary_manifest.json`: contains only **80 rows** (seeds 20261002/03; 40 per seed =
10 prompts × 2 steps × 2 arms). Seed 20261001's 40 rows are missing. Cause verified in
source: `scripts/benchmark_suite.py` skips generation when the PNG already exists
(`if os.path.exists(...): continue`) and then **overwrites** the manifest — the seeds-23
rerun (logs/canary_seeds23.log, Oct 4 04:59) never re-recorded seed-01 rows. The seed-01
images themselves exist and are healthy; only manifest provenance is lost.

Manifest integrity (recomputed independently):
- latent_hash: exactly one per seed in every manifest — canary s02 `1c4b188f…`, s03
  `fa9334cc…` (per-seed regenerated latents, as the script documents); alignment_100
  `976e549d…` = sha256 of the frozen `initial_latent_1024.safetensors` packed latent
  (cross-checks the prior diffusion review's control hash); production_30 `e40e527d…` =
  **recomputed by me** from `initial_latent_2048.safetensors` ([1,16384,64]) — exact match.
- pe_hash: unique per (pid, arm); R0 ≠ O3r for **every** prompt in all three manifests; same
  prompt+arm matches across canary seeds (02 vs 03) for all 10 prompts. I recomputed the
  hash (sha256 of dtype+shape+bytes of the fp32-cast bf16 view, per `common.py::tensor_hash`
  and `benchmark_suite.py`'s exact construction) for 16 sampled (suite, pid, seed, steps,
  arm) combos incl. canary/P/C-series: **16/16 exact matches**.
- gen_s: canary/align100 24.8–48.2 s @1024; production_30 312.9–324.0 s @2048 — sane
  (~10× the 1024 cost for 4× pixels + tiled VAE decode).

**WARNING (W1)**: canary manifest missing seed-01 rows (overwrite/skip defect).
**WARNING (W2)**: `image_metrics_canary___s20261001_40s_R0_vs_O3r.csv` survives with only
**2 rows** (C09, C10; summary n_pairs=2), and the s20261001 **25s** canary CSV has 10 rows
but **no OCR columns** (pre-OCR script version). The reports only cite the seed-1 40s
"subset" LPIPS median 0.089 — which does equal the median of the surviving 2 rows
(0.1535, 0.0246 → 0.089) — but full seed-1 canary 40s medians are no longer reconstructable
from metrics artifacts (images survive; I did not regenerate metrics for the 8 lost pairs).

## 2. EMBEDDING STATS — VERIFIED EXACTLY

Recomputed (float64, from `artifacts/tensors/{R0,O3r}/A0*.npz` `prenorm`) for A001, A037,
A100 — cosine **and** rel-L2 match the per-prompt CSV to <1e-12 (cos 0.9957818720 /
0.9957684386 / 0.9955916771; rel_l2 0.0976042487 / 0.0976518371 / 0.0970344835, reference
norm = first arm R0 — note the CSV's `norm_ratio` ≈ 0.96 reconciles the two rel-L2
conventions). Medians over all 100 CSV rows: cos 0.9957393025, rel_l2 0.0966666345 —
identical to `embedding_summary_stats_R0O3r_O0O3r_R1O0_alignment_100.json`
(claimed cos_med 0.995739 / rel_l2_med 0.0967 ✓; also min cos 0.995390, max rel_l2 0.09914
match the report's "min cos 0.9954", "9.5–9.9%"). Same file's O0_vs_O3r block:
cos med 0.997623 / rel_l2 med 0.06961, max 0.07208 — matches TECHNICAL §4 row exactly.

## 3. IMAGE METRICS — VERIFIED (independent pixel-level recomputation)

`image_metrics_alignment_100___s20261001_40s_R0_vs_O3r.csv` (100 rows): recomputed medians
PSNR **29.2857** / SSIM **0.9680** / LPIPS **0.03027** — claimed 29.286 / 0.968 / 0.0303 ✓
(summary JSON identical). I additionally recomputed 4 rows **from the raw PNGs** with my own
PSNR/SSIM(skimage)/LPIPS(alex) implementations: A001 (47.1699/0.9968/0.0017), A050
(38.6916/0.9923/0.0116), P01 2K (20.9417/0.9265/0.1213), P30 2K (18.9876/0.8641/0.1563) —
**all 12 values match the CSVs to 4 decimals**.

OCR columns:
- align100 40s: **20/100** rows carry cer_a/cer_b (A081–A100, the text-category subset).
  Recomputed deltas (cer_b−cer_a): worse by >0.02 = **0**; better by >0.02 = **6**
  (A085, A086, A090, A095, A097, A099) — claim "0 worse, 6 better" ✓. CER medians
  R0 0.0798 / OV 0.0357 — report's "0.080 vs 0.036" ✓. ("FRESH BAKERY DAILY" C09 exact
  both arms ✓; C10 Chinese CER 0.1 both arms ✓.)
- production_30 2K: **19/30** rows OCR'd. Recomputed: worse 10 (P01,P04,P06,P07,P08,P09,
  P11,P23,P24,P27), tie 4 (P10,P12,P28,P30), better 5 (P02,P03,P05,P25,P29) — claim
  "10 worse / 4 tie / 5 better" ✓. Paired CER median diff **+0.0256** ≈ +0.026 ✓.
  Wilcoxon signed-rank recomputed with scipy (n=19, 4 zero diffs): two-sided p = **0.4180**
  (pratt) / **0.4204** (zsplit) — claimed p=0.42 ✓ (zero-handling convention-dependent;
  both round to 0.42). Combined sign test 10 vs 11 → p = 1.0 ✓.

## 4. REPORT CONSISTENCY — VERIFIED (26+ numbers traced; 4 report-quality defects)

Spot-checks against raw artifacts (all matched unless noted):
1. CUSTOMER §5 / TECHNICAL §4 table: R0:R0R 1.0/0.0; R0:R1 0.998711/0.05283; R1:O0
   1.0/7.7e-6; O0:O1 0.999928/0.01281 (canary stats JSON) ✓.
2. O0:O3 smoke 0.997622/0.0698 and R0:O3 smoke 0.995805/0.0974 — **recomputed by me from
   tensors S01–S03** (medians 0.06980 / 0.09736; cos 0.997622 / 0.995805) ✓, because the
   canary CSV only retains 5 of its 9 filename-claimed pairs (another overwrite casualty —
   numbers themselves correct).
3. O0:O1F ≈1.0/5.7e-05 — recomputed from tensors: median 6e-05 (5/8/6e-05) ✓.
4. LIMITATIONS §4 O3r≈O3 equivalence "7.0/7.0%, 9.7/9.8% on identical prompts": recomputed
   smoke medians O0:O3 0.0698 vs O0:O3r 0.0696*; R0:O3 0.0974 vs R0:O3r 0.0976 ✓.
5. Phase E: 25s cos 0.9971/7.59% and 40s 0.9829/18.49% = trace JSON finals ✓.
6. Phase G crossover: S02 0.0033/0.1852/0.1852/0.0033 and C09 0.0041/0.1948/0.1949/0.0041 —
   exact vs `crossover_S02_k25.json` / `crossover_C09_k25.json` ✓.
7. **Crossover interpretation (prior review's correction)**: EXECUTIVE_SUMMARY #3 ("swapping
   conditioning after step 25 changes the final by only ~0.3–0.4%, swapping the accumulated
   latent state ~19%"), CUSTOMER §9 ("0.3–0.4%" vs "18.5–19.5%", delta locked in during
   early/mid steps), TECHNICAL §9 (explicitly labeled "corrected after independent review";
   notes no-KV-cache regime + A|D 0.18516 vs transplant 0.18490) — **all three state the
   corrected interpretation** (accumulated latent state dominates; late conditioning swap
   washed out). No inverted phrasing remains. ✓
8. Phase H: S02 E1x 18.5%/18.5 dB; randoms 23.5/16.7/17.7% @ 18.0/19.6/19.4 dB; C09 E1x
   18.4%/21.1; randoms 47.2/39.5/32.1% @ 14.3/15.6/16.9 dB; 0.25×delta 1.8%/37 dB (S02:
   0.0178/37.17); "on C09 random deltas 1.7–2.6× larger" (ratios 2.56/2.14/1.74) — all exact
   vs perturbation JSONs ✓.
9. Phase C: entry-0 rel_l2 0.0283 ("~3% at layer 0"), final 0.0685 ("~7%"), final/entry-35
   amplification 0.0685/0.0241 = **2.84×** ("≈2.8×"), zero NaN/Inf ✓. Minor: "1.0–1.4e-2
   through the trunk" — actual entries 1–34 span **0.88–1.6e-2** (rounding-level, W6).
10. align100 25s medians 28.8/0.967/0.036 (p5 18.9/p95 43.2/0.212) and 40s 29.3/0.968/0.030
    (18.9/42.6/0.159) ✓; 2K §12: 27.8/0.964/0.032 (p5 19.8/p95 40.9) ✓ — matches
    `image_metrics_production_30___*summary.json` exactly.
11. Canary: C01 25s PSNR 42.6 / C02 20.8 ✓; 3-seed 40s LPIPS medians 0.0409/0.0508 vs 0.089
    (seed-1 n=2 subset) ✓.
12. CPU perf table: O0 34.69/0.834/1.074/2.492 (p95 2.516); O1 12.56/0.387/0.919/1.330
    (1.372); O3r 1.41/0.218/0.711/1.080 (1.208) — exact vs `cpu_performance.json` ✓ (but see
    W4).
13. LIMITATIONS.md documents all four required items: 1-seed 100-suite deviation (§7),
    2K-suite seed scope (§8), O3 artifact host IR bug + O3r reproduction rationale (§4),
    blind evaluation PENDING / nothing fabricated (§10) ✓.

Report-quality defects found:
- **W3**: EXECUTIVE_SUMMARY key-numbers table lists "n/a" for O0-vs-O3 and R0-vs-O3 (and
  twice in Answer #2) although the measurements exist and appear in the other two reports
  (7.0% and 9.7%). Placeholder left unfilled.
- **W4**: TECHNICAL §12 captions the CPU table "accuracy configs", but `cpu_performance.json`
  records `EXECUTION_MODE_HINT: "PERFORMANCE"` for all three arms (INFERENCE_PRECISION
  f32/bf16 are as claimed). Performance-only impact; contradicts the caption and §6's
  "verified via runtime properties" (which holds for the O0 encoder runs, not this bench).
- **W6**: trunk band understated (above).
- **W8**: LIMITATIONS §8 points to the prod-30 manifest for the seed-deviation record; that
  JSON contains only rows/total_s (the deviation text actually lives in CUSTOMER §12).

## 5. GALLERY — VERIFIED with one gap (W5)

36 PNGs, all valid, 1024² or 2048², 30–3000+ KB (none trivial); spot-checked two are
byte-identical (md5) to their `artifacts/images/` sources. Categories present:
best_aligned, median, worst_divergence, text_english, text_chinese, multiseed_canary,
2k_best_aligned, 2k_median, 2k_poster, 2k_text_english, 2k_text_chinese, 2k_worst_ocr_pair,
with INDEX.md. **Missing**: `crossover_*` and `perturbation_*` categories (expected per the
review brief; the underlying images exist in artifacts/images/crossover_* and pert_*).
Gallery selections are sensible (e.g. worst_divergence = C02, the 20.8 dB case).

## 6. BLIND PACKAGE — VERIFIED

`reports/blind_package/index.json`: 100 pairs (`pair_000..099`, alignment_100 40s tags),
instructions present, `evaluation_status: "PENDING — no human results fabricated"` ✓.
200 PNGs (100×left/right) exist. Key at `artifacts/manifests/blind_package_KEY.json` maps
every pair to left/right arms; randomization genuine (R0 left 49 / O3r left 51) ✓.

## 7. FIGURES / PLOTS — VERIFIED (programmatic)

`artifacts/plots/` contains all 13 required categories (28 PNGs): embedding_histogram,
embedding_error_heatmap, layer_cosine_vs_depth_{R0_vs_R1,R1_vs_O0,R0_vs_O3,R0_vs_O3r,
O0_vs_O3,O0_vs_O3r}, layer_relative_l2_vs_depth_* (same 6), layer_max_abs_* (bonus),
diffusion_latent_divergence_{25,40}steps, diffusion_model_output_{25,40}steps,
steps_25_vs_40_comparison, perturbation_vs_final_divergence, precision_chain,
ov_accuracy_performance_tradeoff, suite100_quality_distributions, text_rendering_accuracy.
All non-zero (32–116 KB), valid PNG headers. NOTE: I could not visually inspect the plots —
this review model does not support image input; attempted image reads returned
"Cannot read image". Substitute programmatic checks on the two designated plots:
layer_relative_l2_vs_depth_R0_vs_O3r.png (980×700, pixel std 28.6, full 256-level range,
5.2% non-white — rendered curve content, not blank) and suite100_quality_distributions.png
(1540×588, std 50.9, 20.1% non-white — multi-panel distribution content). Rendering is
sensible by these statistics; human visual confirmation was not possible.

---

## Discrepancy register

- **D1/W1 (bookkeeping)**: `suite_canary_manifest.json` holds 80/120 rows — seed-20261001
  rows lost to the skip-existing+overwrite design of `benchmark_suite.py` (proven from
  source + file timestamps). Provenance loss only; images and all other manifests intact.
- **D2/W2 (bookkeeping)**: canary seed-01 40s metrics CSV truncated to 2 rows and its 25s
  CSV predates OCR columns; the canary embedding CSV retains 5/9 pairs (smoke pair numbers
  still verified by my tensor recomputation). Third instance of the overwrite defect family
  already flagged by both prior reviews.
- **D3/W3 (report)**: executive summary "n/a" placeholders for INT8 deltas (data exists).
- **D4/W4 (report)**: "accuracy configs" caption vs PERFORMANCE execution mode in the CPU
  benchmark artifact (perf-only; precision hints correct).
- **D5/W5 (gallery)**: crossover/perturbation categories absent.
- **D6/W6 (rounding)**: trunk layer band 0.88–1.6e-2 quoted as 1.0–1.4e-2.
- **D7/W7 (scope note)**: OCR coverage is a designed subset (20/100 text prompts; 19/30 at
  2K); claims are correctly scoped to it in the reports.
- **D8/W8 (pointer)**: LIMITATIONS §8's "documented there" manifest pointer is wrong (the
  deviation text is in CUSTOMER §12; manifest has no such field).
- **Note**: Wilcoxon p=0.42 reproduces only with pratt/zsplit zero handling (0.418/0.420);
  default wilcox gives 0.57 — the report value is defensible but convention-dependent.

## Bottom line

Suite coverage, manifest hash discipline (16/16 recomputed), embedding statistics (exact),
image metrics (pixel-level recomputation of medians and rows), OCR claims, Wilcoxon/sign
tests, crossover/perturbation tables, 2K section, limitations, blind package, and all 13
figure categories verify against raw artifacts, and the corrected crossover interpretation
is consistently stated in all narrative reports. Defects are confined to overwritten
canary-era bookkeeping files, unfilled executive-summary placeholders, a mis-captioned
performance table, and gallery category gaps — none affect the scientific conclusions.
