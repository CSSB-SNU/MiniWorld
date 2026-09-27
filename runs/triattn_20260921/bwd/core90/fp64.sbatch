#!/bin/bash
#SBATCH -J mw-dq-fp64
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90/fp64-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 64 128; do
  for MASK in none mixed one_key all_masked; do
    bash "$ENV" python -u -B "$R/dq/check.py" --length "$L" --mask "$MASK" --no-bench --output "$R/core90/fp64-${SLURM_JOB_ID}-L${L}-${MASK}.json"
  done
done
