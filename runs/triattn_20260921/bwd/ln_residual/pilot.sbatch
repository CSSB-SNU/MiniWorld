#!/bin/bash
#SBATCH -J mw-ln-residual
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/ln_residual/pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/ln_residual
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 8 16 64 384 768 1024; do
 bash "$ENV" python -u -B "$R/check.py" --length "$L" --output "$R/pilot-${SLURM_JOB_ID}-L${L}.json"
done
