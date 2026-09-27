#!/bin/bash
#SBATCH -J mw-bwd-sol
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/sol-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for VARIANT in q32_coop q32_coop_stats q32_coop_pack; do
  for L in 64 768 1024; do
    FUSION_ARTIFACT="$VARIANT" FUSION_ROW_OPTIONS=4 bash "$ENV" python -u -B "$R/bench_sol.py" --length "$L" --output "$R/sol-${SLURM_JOB_ID}-${VARIANT}-L${L}.json"
  done
done
