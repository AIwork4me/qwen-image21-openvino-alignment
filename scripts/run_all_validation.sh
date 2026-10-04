#!/usr/bin/env bash
# Full validation pipeline. Stages are checkpointed; artifacts accumulate under artifacts/.
set -x
cd "$(dirname "$0")/.."

python scripts/collect_environment.py
bash scripts/00_install_diffusers_qi21.sh

for s in smoke canary alignment_100 production_30; do
  python scripts/prepare_inputs.py --suite $s
done
python scripts/compare_inputs.py smoke
python -m tests.test_conditioning_semantics

python scripts/run_rocm_encoder.py --arm R0  --suite smoke
python scripts/run_rocm_encoder.py --arm R0R --suite smoke --repeats 3
python scripts/run_rocm_encoder.py --arm R1  --suite smoke
python scripts/export_ov_clean.py
python scripts/export_ov_int8.py
python scripts/run_openvino_encoder.py --arm O0  --suite smoke
python scripts/run_openvino_encoder.py --arm O1  --suite smoke --runtime-precision bf16
python scripts/run_openvino_encoder.py --arm O3  --suite smoke
python scripts/run_openvino_encoder.py --arm O1F --suite smoke
python scripts/run_openvino_encoder.py --arm O3r --suite smoke

python scripts/compare_embeddings.py --pairs R0:R0R R0:R1 R1:O0 R0:O0 O0:O1 O0:O1F O0:O3 O0:O3r R0:O3 R0:O3r --suite smoke
python scripts/compare_layers.py --ref R1 --target O0 --suite smoke
python scripts/compare_layers.py --ref O0 --target O3 --suite smoke
python scripts/compare_layers.py --ref R0 --target R1 --suite smoke
python scripts/compare_layers.py --ref R0 --target O3 --suite smoke

# canary + 100 suites
for arm in R0 R0R R1; do python scripts/run_rocm_encoder.py --arm $arm --suite canary; done
for arm in O0 O1 O3r; do python scripts/run_openvino_encoder.py --arm $arm --suite canary; done
python scripts/run_openvino_encoder.py --arm O0  --suite alignment_100
python scripts/run_openvino_encoder.py --arm O1  --suite alignment_100 --runtime-precision bf16
python scripts/run_openvino_encoder.py --arm O3r --suite alignment_100
python scripts/run_rocm_encoder.py --arm R0 --suite alignment_100

python scripts/generate_fixed_latents.py
python scripts/run_diffusion_trace.py --suite smoke --pid S01 --steps 25 --arm-a R0 --arm-b R0 --seed-tag S01_25s_selfcheck
python scripts/run_diffusion_trace.py --suite smoke --pid S02 --steps 25 --arm-a R0 --arm-b O3 --save-tensors
python scripts/run_diffusion_trace.py --suite smoke --pid S02 --steps 40 --arm-a R0 --arm-b O3 --save-tensors
python scripts/crossover_25_40.py --suite smoke --pid S02 --steps 40 --k 25 --arm-a R0 --arm-b O3
python scripts/crossover_25_40.py --suite canary --pid C09 --steps 40 --k 25 --arm-a R0 --arm-b O3r
python scripts/perturbation_sweep.py --suite smoke --pid S02 --arm-ref R0 --arm-ov O3  --resolutions-steps 1024:25,1024:40
python scripts/perturbation_sweep.py --suite canary --pid C09 --arm-ref R0 --arm-ov O3r --resolutions-steps 1024:25,1024:40

python scripts/benchmark_suite.py --suite canary        --arms R0,O3r --steps 25,40 --seeds 20261001
python scripts/benchmark_suite.py --suite alignment_100 --arms R0,O3r --steps 25,40 --seeds 20261001
python scripts/benchmark_suite.py --suite production_30 --arms R0,O3r --steps 40    --seeds 20261001 --resolution 2048

python scripts/image_metrics.py --pattern "canary_*_s20261001_25s" --arm-a R0 --arm-b O3r
python scripts/image_metrics.py --pattern "canary_*_s20261001_40s" --arm-a R0 --arm-b O3r --ocr
python scripts/image_metrics.py --pattern "alignment_100_*_s20261001_25s" --arm-a R0 --arm-b O3r --ocr
python scripts/image_metrics.py --pattern "alignment_100_*_s20261001_40s" --arm-a R0 --arm-b O3r --ocr
python scripts/image_metrics.py --pattern "production_30_*_s20261001_40s" --arm-a R0 --arm-b O3r --ocr

python scripts/benchmark_cpu.py --arms O0,O1,O3r,O3 --n-warm 3 --n-iter 10
python scripts/blind_package.py --arm-a R0 --arm-b O3r --pattern "alignment_100_*_s20261001_40s"
python scripts/build_model_manifest.py
python scripts/verify_artifacts.py
python scripts/build_report.py
echo RUN_ALL_VALIDATION_COMPLETE
