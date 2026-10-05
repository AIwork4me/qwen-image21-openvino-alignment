#!/usr/bin/env bash
# Tier 3: trajectories + crossover + perturbation (frozen latent controls).
set -euo pipefail
cd "$(dirname "$0")/.."
export V2_VALDATA=${V2_VALDATA:-/valdata}
source /workspace/qwen-image21-v2-venv/bin/activate
python scripts/v2_check_environment.py
python scripts/v2_trajectory.py --suite smoke --pids S01,S02 --steps 25 40
python scripts/v2_crossover.py --pid C09 --steps 40
python scripts/v2_crossover.py --pid S02 --steps 40
python scripts/v2_perturbation.py --pid C09
python scripts/v2_perturbation.py --pid S02
echo "TIER3 TRAJECTORY COMPLETE"
