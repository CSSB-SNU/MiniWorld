#!/bin/bash
#SBATCH -J mw-bias-shared
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/shared-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for V in q32_shared q64_shared; do
  for L in 64 768; do
    FUSION_ARTIFACT="$V" bash "$ENV" python -u -B "$R/bench.py" --length "$L" --output "$R/shared-${SLURM_JOB_ID}-${V}-L${L}.json"
  done
done
