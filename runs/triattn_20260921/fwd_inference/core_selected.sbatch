#!/bin/bash
#SBATCH -J mw-triattn-coreselected
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/core-selected-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for CELL in 384:h4kv_local1:h4kv_local1 768:hot6t:hot6t 1024:qfull4v2:hot4t; do
  IFS=: read -r L CORE BASE <<< "$CELL"
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --default-entry --selected --front --front-artifact front8 --qualify --artifact "$CORE" --length "$L" --rounds 24 --replays 30 --output "$R/core-selected-module-${L}-${SLURM_JOB_ID}.json"
  timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --default-entry --selected --artifact "$CORE" --previous --previous-artifact "$BASE" --previous-out-artifact outproj_c1 --length "$L" --rounds 64 --replays 40 --output "$R/core-selected-anthropic-${L}-${SLURM_JOB_ID}.json"
done
