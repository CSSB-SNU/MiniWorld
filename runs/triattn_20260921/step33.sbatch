#!/bin/bash
#SBATCH -J mw-ta-final
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step33-%j.out
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
hostname
for E in 0 1; do
 for L in 384 768 1024; do
  ENDING=$E bash "$ENV" python -u -B "$R/qualify_final.py" --length "$L" --output "$R/logs/step33-e${E}-L${L}.json"
 done
done
if grep -q 'refused by kernels.triattn' "$R/logs/step33-${SLURM_JOB_ID}.out"; then exit 1; fi
