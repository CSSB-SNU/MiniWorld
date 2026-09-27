#!/bin/bash
#SBATCH -J mw-ta-projected-bench
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected/fused-bench-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python "$R/screen_fused.py" "${PROJECTED_BUILD_LOG:?}"
timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench_fused.py" --length 768 --output "$R/bench-${PROJECTED_VARIANT}-${SLURM_JOB_ID}.json"
