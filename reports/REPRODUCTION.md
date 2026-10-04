# Exact reproduction instructions (clean environment)

Everything below was executed on:
AMD EPYC 9334 (Zen4, AVX512-BF16/VNNI) + AMD Radeon Pro W7900D (gfx1100),
ROCm userspace 7.14.0 (therock dist), torch 2.13.0+rocm7.14.0, Python 3.12.

## 0. Environment

```bash
bash scripts/collect_environment.py          # captures system/ROCm/python metadata
```

ROCm 7.14.0 userspace correction performed once (reversible):
```bash
mkdir -p /opt/rocm-7.14.0
tar -xzf /workspace/therock-dist-linux-gfx110X-all-7.14.0.tar.gz -C /opt/rocm-7.14.0
ln -sfn /opt/rocm-7.14.0 /etc/alternatives/rocm && ldconfig
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Diffusers: 0.40.0 + vendored Qwen-Image-2.1 modules from diffusers main
commit 8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd (files listed in
requirements-lock.txt; applied by scripts/00_install_diffusers_qi21.sh).

## 1. Model freeze

```bash
# Qwen/Qwen-Image-2.1 @ snapshot d26bb61231c349cf6b7896fa83353113880e1ba3
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Image-2.1", max_workers=8)
PY
python scripts/build_model_manifest.py        # hashes configs + weights + OV artifacts
```

## 2. Inputs (Phase A)

```bash
python scripts/prepare_inputs.py --suite smoke
python scripts/prepare_inputs.py --suite canary
python scripts/prepare_inputs.py --suite alignment_100
python scripts/prepare_inputs.py --suite production_30
python scripts/compare_inputs.py smoke        # processor vs ComfyUI qwen25 tokenizer
python -m tests.test_conditioning_semantics   # bit-parity with official pipeline
```

## 3. Encoder arms (Phases B/C/D)

```bash
# ROCm reference arms (GPU)
python scripts/run_rocm_encoder.py --arm R0  --suite canary
python scripts/run_rocm_encoder.py --arm R0R --suite canary --repeats 3
python scripts/run_rocm_encoder.py --arm R1  --suite canary
# OpenVINO arms (CPU)
python scripts/export_ov_clean.py                            # O0 clean FP32 export (~6 min)
python scripts/export_ov_int8.py                             # O3r INT8 reproduction
python scripts/run_openvino_encoder.py --arm O0  --suite canary
python scripts/run_openvino_encoder.py --arm O1  --suite canary --runtime-precision bf16
python scripts/run_openvino_encoder.py --arm O3  --suite smoke   # customer artifact (host-limited)
python scripts/run_openvino_encoder.py --arm O3r --suite canary  # INT8 reproduction, full coverage
# Comparisons
python scripts/compare_embeddings.py --pairs R0:R0R R0:R1 R1:O0 R0:O0 O0:O1 O0:O3 O0:O3r R0:O3 R0:O3r --suite canary
python scripts/compare_layers.py --ref R1 --target O0 --suite canary
python scripts/compare_layers.py --ref O0 --target O3 --suite smoke
python scripts/compare_layers.py --ref O0 --target O3r --suite canary
python scripts/compare_layers.py --ref R0 --target R1 --suite canary
python scripts/compare_layers.py --ref R0 --target O3r --suite canary
```

## 4. Diffusion controls + transplant + traces (Phases E/F)

```bash
python scripts/generate_fixed_latents.py
python scripts/run_diffusion_trace.py --suite smoke --pid S01 --steps 25 --arm-a R0 --arm-b R0 --seed-tag S01_25s_selfcheck
python scripts/run_diffusion_trace.py --suite smoke --pid S02 --steps 25 --arm-a R0 --arm-b O3 --save-tensors
python scripts/run_diffusion_trace.py --suite smoke --pid S02 --steps 40 --arm-a R0 --arm-b O3 --save-tensors
```

## 5. Crossover + perturbation (Phases G/H)

```bash
python scripts/crossover_25_40.py --suite smoke --pid S02 --steps 40 --k 25 --arm-a R0 --arm-b O3
python scripts/crossover_25_40.py --suite canary --pid C09 --steps 40 --k 25 --arm-a R0 --arm-b O3r
python scripts/perturbation_sweep.py --suite smoke --pid S02 --arm-ref R0 --arm-ov O3  --resolutions-steps 1024:25,1024:40
python scripts/perturbation_sweep.py --suite canary --pid C09 --arm-ref R0 --arm-ov O3r --resolutions-steps 1024:25,1024:40
```

## 6. Suites (Phases I/J)

```bash
python scripts/benchmark_suite.py --suite canary       --arms R0,O3r --steps 25,40 --seeds 20261001
python scripts/benchmark_suite.py --suite alignment_100 --arms R0,O3r --steps 25,40 --seeds 20261001
python scripts/benchmark_suite.py --suite production_30 --arms R0,O3r --steps 40    --seeds <seeds> --resolution 2048
python scripts/image_metrics.py --pattern "alignment_100_*_s20261001_40s" --arm-a R0 --arm-b O3r --ocr
python scripts/blind_package.py --arm-a R0 --arm-b O3r --pattern "alignment_100_*_s20261001_40s"
```

## 7. Performance (CPU) — kept separate from accuracy conclusions

```bash
python scripts/benchmark_cpu.py --arms O0,O1,O3r,O3 --n-warm 3 --n-iter 10
```

## 8. Integrity + reports

```bash
python scripts/verify_artifacts.py
python scripts/build_report.py
```

## Everything

```bash
bash run_all_validation.sh
```
