#!/bin/bash
#SBATCH -J mw-triattn-head4kv
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/head4-kv-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for ARTIFACT in ${ARTIFACTS:-h4kv_stream1 h4kv_stream2 h4kv_stream4 h4kv_resident6}; do
  KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$ARTIFACT"
  bash "$ENV" cuobjdump --dump-resource-usage "$R/$ARTIFACT/build/triattn_infer_${ARTIFACT}.so" > "$R/$ARTIFACT/resources.txt"
  for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/check-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
  for L in 384 768 1024; do
    timeout 300s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact "$ARTIFACT" --previous --previous-artifact hot6t --length "$L" --rounds 24 --replays 30 --output "$R/anthropic-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
