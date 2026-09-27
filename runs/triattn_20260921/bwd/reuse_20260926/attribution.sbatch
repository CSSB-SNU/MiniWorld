#!/bin/bash
#SBATCH -J mw-bwd-reuse-base
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926/attribution-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 768 1024 384; do
  timeout 240s bash "$ENV" python -u -B "$R/attribution.py" --length "$L" --output "$R/attribution-${L}-${SLURM_JOB_ID}.json"
done
