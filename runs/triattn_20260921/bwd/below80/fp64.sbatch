#!/bin/bash
#SBATCH -J mw-below80-fp64
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80/fp64-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 8 16 64; do
  LN_ARTIFACT="$LN_ARTIFACT" bash "$ENV" python -u -B "$R/../ln_residual/check.py" --length "$L" --output "$R/ln-fp64-${SLURM_JOB_ID}-L${L}.json"
done
for L in 64 128; do
  for MASK in none mixed one_key all_masked; do
    DQ_ARTIFACT="$DQ_ARTIFACT" bash "$ENV" python -u -B "$R/../dq/check.py" --length "$L" --mask "$MASK" --no-bench --output "$R/dq-fp64-${SLURM_JOB_ID}-L${L}-${MASK}.json"
  done
done
