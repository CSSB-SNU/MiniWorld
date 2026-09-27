#!/bin/bash
#SBATCH -J mw-triattn-algo-finish
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/algo-finish-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
if [[ -f "$R/anthropic-hot6t-1024-18586.json" ]]; then
  echo 'Original qualification reached final benchmark; inspect before rerunning.'
  exit 0
fi
for CASE in mixed all_masked large_logits; do
  timeout 600s bash "$ENV" compute-sanitizer --tool racecheck --kernel-name 'regex=qkv_attention_inference' --error-exitcode 86 python -u -B "$R/check.py" --artifact hot6t --length 1024 --native-only --cases "$CASE" --output "$R/racecheck-hot6t-1024-${CASE}-${SLURM_JOB_ID}.json"
done
timeout 360s bash "$ENV" compute-sanitizer --tool synccheck --kernel-name 'regex=qkv_attention_inference' --error-exitcode 86 python -u -B "$R/check.py" --artifact hot6t --length 1024 --native-only --cases mixed all_masked large_logits --output "$R/synccheck-hot6t-1024-${SLURM_JOB_ID}.json"
for L in 384 768 1024; do
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact front8 --qualify --artifact hot6t --length "$L" --rounds 24 --replays 30 --output "$R/qualified-hot6t-${L}-${SLURM_JOB_ID}.json"
  timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact hot6t --previous --length "$L" --rounds 64 --replays 40 --output "$R/anthropic-hot6t-${L}-${SLURM_JOB_ID}.json"
done
