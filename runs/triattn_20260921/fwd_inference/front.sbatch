#!/bin/bash
#SBATCH -J mw-triattn-infer-front
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/front-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact front
timeout 180s bash "$ENV" python -u -B "$R/front_check.py" --length 128 --output "$R/front-check-128-${SLURM_JOB_ID}.json"
for ARTIFACT in resident6 stream4; do
  for L in 384 768 1024; do
    timeout 240s bash "$ENV" python -u -B "$R/module_check.py" --front --artifact "$ARTIFACT" --length "$L" --output "$R/front-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
