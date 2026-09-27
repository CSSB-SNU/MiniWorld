#!/bin/bash
#SBATCH -J mw-triattn-wide-bias16-check
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/bias-check-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for D in 64 128; do
 for L in 64 128; do
  for MASK in none mixed one_key all_masked; do
   timeout 90s bash "$ENV" python -u -B "$R/check_bias.py" --head-dim "$D" --length "$L" --mask "$MASK" --output "$R/check-bias-h${D}-L${L}-${MASK}-${SLURM_JOB_ID}.json"
  done
 done
done
