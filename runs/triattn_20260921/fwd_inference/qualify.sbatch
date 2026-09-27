#!/bin/bash
#SBATCH -J mw-triattn-infer-qualify
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
FRONT=${FRONT:-front}
for L in 128 384 768 1024; do
  timeout 180s bash "$ENV" python -u -B "$R/front_check.py" --artifact "$FRONT" --length "$L" --output "$R/${FRONT}-check-${L}-${SLURM_JOB_ID}.json"
done
for TOOL in memcheck racecheck synccheck; do
  timeout 300s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=inference_' --error-exitcode 86 python -u -B "$R/front_check.py" --artifact "$FRONT" --length 1024 --native-only --output "$R/${FRONT}-${TOOL}-${SLURM_JOB_ID}.json"
done
for L in 384 768 1024; do
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact "$FRONT" --qualify --profile --artifact resident6 --length "$L" --rounds 64 --replays 40 --output "$R/qualified-${FRONT}-resident6-${L}-${SLURM_JOB_ID}.json"
done
