#!/bin/bash
#SBATCH -J mw-head4-front-check
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/head4-front-check-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
ARTIFACT=${ARTIFACT:-h4kv_front1}
for L in ${LENGTHS:-128 384 768 1024}; do
  timeout 240s bash "$ENV" python -u -B "$R/head4_front_check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/frontcheck-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  for TOOL in memcheck racecheck synccheck; do
    timeout 600s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=front_head4_kv_kernel' --error-exitcode 86 python -u -B "$R/head4_front_check.py" --artifact "$ARTIFACT" --length "$L" --native-only --output "$R/front-${TOOL}-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
