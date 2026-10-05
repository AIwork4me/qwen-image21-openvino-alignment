# Qwen-Image 2.1 — OpenVINO(CPU) vs ROCm(GPU) Text Encoder Alignment Validation

> **结论、证据与复现入口：[CONCLUSIONS.md](CONCLUSIONS.md)**

Customer-grade, reproducible investigation of whether the OpenVINO Qwen3-VL-8B text
encoder is numerically aligned with the AMD GPU (W7900D / gfx1100 / ROCm 7.14.0)
PyTorch reference for Qwen-Image 2.1, and why 25-step images look similar while
40-step images can differ.

## One-command reproduction

```bash
bash scripts/run_all_validation.sh      # full pipeline (long: many GPU-hours)
```

Stages are checkpointed; see REPRODUCTION.md for per-stage commands.

## Layout

- `config/` experiment.yaml (paths, arms, seeds), thresholds.yaml (empirical bands)
- `prompts/` fixed prompt suites (smoke/canary/alignment_100/production_30)
- `scripts/` all experiment drivers (see REPRODUCTION.md)
- `artifacts/` raw evidence (tensors, images, metrics, logs, manifests, reviews)
- `reports/` customer/technical/executive reports + gallery
- `tests/` conditioning-semantics unit tests

## Headline results (see reports/)

- Tokenization: EXACT identity between official processor path and customer path.
- OpenVINO FP32 (clean export): rel L2 ~8e-6 vs PyTorch FP32 — numerically aligned.
- Customer INT8 artifact: rel L2 ~7% vs OV FP32 — quantization noise, same order as
  the GPU's own bf16-vs-fp32 envelope (~5.3%).
- 40-step divergence is accumulated trajectory divergence (crossover experiment),
  and generic: random embedding deltas of equal magnitude cause equal or larger
  image differences (perturbation experiment).
- No measurable quality/text-rendering regression on canary suite.
