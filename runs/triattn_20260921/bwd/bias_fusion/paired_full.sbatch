#!/bin/bash
#SBATCH -J mw-bias-full-pair
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/paired-full-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
  bash "$ENV" python -u -B "$R/paired_full.py" --length "$L" --output "$R/paired-full-${SLURM_JOB_ID}-L${L}.json"
done
