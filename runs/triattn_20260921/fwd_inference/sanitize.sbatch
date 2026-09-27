#!/bin/bash
#SBATCH -J mw-triattn-infer-sanitize
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/sanitize-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 128 384 768 1024; do
  for TOOL in memcheck racecheck synccheck; do
    timeout 480s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=qkv_attention_inference' --error-exitcode 86 python -u -B "$R/check.py" --artifact "${ARTIFACT:-resident6}" --length "$L" --native-only --cases mixed all_masked --output "$R/${TOOL}-${ARTIFACT:-resident6}-${L}-${SLURM_JOB_ID}.json"
  done
done
