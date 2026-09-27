#!/bin/bash
#SBATCH -J mw-triattn-infer-module
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/module-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for ARTIFACT in resident6 resident4; do
  for L in 384 768 1024; do
    timeout 240s bash "$ENV" python -u -B "$R/module_check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/module-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
