#!/bin/bash
#SBATCH -J mw-ta-projected-stsm
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected/stsm-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
if [ "${SKIP_SCALAR:-0}" != 1 ]; then
 bash "$ENV" python -u "$R/build.py" > "$R/build-scalar-${SLURM_JOB_ID}.log" 2>&1
 bash "$ENV" python -u "$R/check.py" --variant proof --output "$R/check-scalar-${SLURM_JOB_ID}.json"
fi
bash "$ENV" python -u "$R/build_stsm.py" > "$R/build-stsm-${SLURM_JOB_ID}.log" 2>&1
bash "$ENV" python -u "$R/check.py" --variant stsm --output "$R/check-stsm-${SLURM_JOB_ID}.json"
