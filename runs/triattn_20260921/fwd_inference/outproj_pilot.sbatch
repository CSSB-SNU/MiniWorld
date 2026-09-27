#!/bin/bash
#SBATCH -J mw-triattn-outproj
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/outproj-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in ${ARTIFACTS:-outproj_c1 outproj_c2 outproj_c4}; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$A"
  bash "$ENV" cuobjdump --dump-resource-usage "$R/$A/build/triattn_infer_${A}.so" > "$R/$A/resources.txt"
  for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/outproj_check.py" --artifact "$A" --length "$L" --output "$R/check-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for CELL in 384:h4kv_local1 768:hot6t 1024:hot4t; do
    L=${CELL%%:*}; CORE=${CELL#*:}
    timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --selected --artifact "$CORE" --out-artifact "$A" --previous --previous-artifact selected --length "$L" --rounds 32 --replays 40 --output "$R/anthropic-${A}-${L}-${SLURM_JOB_ID}.json"
  done
done
