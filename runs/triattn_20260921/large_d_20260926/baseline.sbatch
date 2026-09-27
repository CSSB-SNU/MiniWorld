#!/bin/bash
#SBATCH -J mw-triattn-wide-baseline
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/baseline-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for C in 256 512; do
  for L in 384 768; do
    timeout 450s bash "$ENV" python -u -B "$R/baseline.py" --width "$C" --length "$L" --output "$R/baseline-C${C}-L${L}-${SLURM_JOB_ID}.json"
  done
done
