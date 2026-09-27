#!/bin/bash
#SBATCH -J mw-triattn-outqual
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/outproj-qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${ARTIFACT:?output artifact required}"
for L in 128 384 768 1024; do
  for TOOL in memcheck racecheck synccheck; do
    timeout 600s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=inference_output_residual' --error-exitcode 86 python -u -B "$R/outproj_check.py" --artifact "$ARTIFACT" --length "$L" --native-only --output "$R/${TOOL}-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
for CELL in ${CELLS:-384:h4kv_local1 768:hot6t 1024:hot4t}; do
  L=${CELL%%:*}; CORE=${CELL#*:}
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --selected --front --front-artifact front8 --qualify --artifact "$CORE" --out-artifact "$ARTIFACT" --length "$L" --rounds 24 --replays 30 --output "$R/qualified-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --selected --artifact "$CORE" --out-artifact "$ARTIFACT" --previous --previous-artifact selected --length "$L" --rounds 64 --replays 40 --output "$R/anthropic-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
