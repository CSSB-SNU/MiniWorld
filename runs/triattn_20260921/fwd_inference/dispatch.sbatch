#!/bin/bash
#SBATCH -J mw-triattn-infer-dispatch
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:08:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/dispatch-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
  timeout 120s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact front8 --profile --profile-only --artifact resident6 --length "$L" --output "$R/dispatch-front8-resident6-${L}-${SLURM_JOB_ID}.json"
done
