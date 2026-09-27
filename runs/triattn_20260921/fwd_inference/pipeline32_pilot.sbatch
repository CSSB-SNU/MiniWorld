#!/bin/bash
#SBATCH -J mw-triattn-pipe32
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/pipeline32-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in ${ARTIFACTS:-pipe32_c4 serial32_c4 pipe32_c2}; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$A"
  bash "$ENV" cuobjdump --dump-resource-usage "$R/$A/build/triattn_infer_${A}.so" > "$R/$A/resources.txt"
  for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check.py" --artifact "$A" --length "$L" --output "$R/check-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for L in 64 128 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/hot_audit.py" --artifact "$A" --length "$L" --output "$R/audit-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for L in 384 768 1024; do
    timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact "$A" --previous --previous-artifact selected --length "$L" --rounds 24 --replays 30 --output "$R/anthropic-${A}-${L}-${SLURM_JOB_ID}.json"
  done
done
