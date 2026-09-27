#!/bin/bash
#SBATCH -J mw-triattn-infer-front-tune
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/front-tune-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for FRONT in front16 front8; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$FRONT"
  timeout 180s bash "$ENV" python -u -B "$R/front_check.py" --artifact "$FRONT" --length 384 --output "$R/check-${FRONT}-${SLURM_JOB_ID}.json"
  for L in 384 768 1024; do
    timeout 240s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact "$FRONT" --artifact resident6 --length "$L" --output "$R/${FRONT}-resident6-${L}-${SLURM_JOB_ID}.json"
  done
done
