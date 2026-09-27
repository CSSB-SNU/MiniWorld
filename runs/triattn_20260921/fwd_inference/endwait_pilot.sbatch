#!/bin/bash
#SBATCH -J mw-triattn-endwait
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/endwait-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in ${ARTIFACTS:-pipeend32_c4 pipeend64_c2 pipeend64_c4}; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$A"
  bash "$ENV" cuobjdump --dump-resource-usage "$R/$A/build/triattn_infer_${A}.so" > "$R/$A/resources.txt"
  bash "$ENV" cuobjdump --dump-sass "$R/$A/build/triattn_infer_${A}.so" > "$R/$A/sass.txt"
  for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check.py" --artifact "$A" --length "$L" --output "$R/check-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for L in 64 128 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/hot_audit.py" --artifact "$A" --length "$L" --output "$R/audit-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for L in 384 768 1024; do
    timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact "$A" --out-artifact outproj_c1 --previous --previous-artifact selected --previous-out-artifact outproj_c1 --length "$L" --rounds 24 --replays 30 --output "$R/anthropic-${A}-${L}-${SLURM_JOB_ID}.json"
  done
done
