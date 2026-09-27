#!/bin/bash
#SBATCH -J mw-triattn-infer-build
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/build-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for ARTIFACT in resident6 resident4; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$ARTIFACT"
  bash "$ENV" cuobjdump --dump-resource-usage "$R/$ARTIFACT/build/triattn_infer_${ARTIFACT}.so" > "$R/$ARTIFACT/resources.txt"
  bash "$ENV" cuobjdump --dump-sass "$R/$ARTIFACT/build/triattn_infer_${ARTIFACT}.so" > "$R/$ARTIFACT/sass.txt"
  for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/check-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
