# Historical: prior v2 attempt with pip ROCm runtime 7.14.0

A previous v2 session on this machine installed the mandated torch
2.12.0+rocm7.14.1 / torchvision 0.27.0+rocm7.14.1 wheels but then
downgraded the pip ROCm SDK/runtime packages to 7.14.0 (log:
rocm7140_restore_attempt.log). That state produced the warnings captured in
torch_gpu_stability_extended.txt / torch_gpu_hard_validation.txt
("compiled against 7.14.1 but the installed ROCm version is 7.14.0").

That mixed environment violated the v2 authoritative stack
(ROCm 7.14.1). Its numerical artifacts were regenerated from scratch in a
clean, fully-conforming ROCm 7.14.1 environment (see parent directory and
scripts/setup_v2_rocm7141_w7900.sh). These files are preserved unchanged as
historical evidence of the deviation only; they are NOT v2 environment state.
