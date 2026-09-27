#!/bin/bash
#SBATCH -J mw-triattn-algo-qualify
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/algo-qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
ARTIFACT=${ARTIFACT:-hot4}
for L in 64 128 384 768 1024; do
  timeout 240s bash "$ENV" python -u -B "$R/hot_audit.py" --artifact "$ARTIFACT" --length "$L" --output "$R/audit-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
for L in 128 384 768 1024; do
  for TOOL in memcheck racecheck synccheck; do
    timeout 1200s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=qkv_attention_inference' --error-exitcode 86 python -u -B "$R/check.py" --artifact "$ARTIFACT" --length "$L" --native-only --cases mixed all_masked large_logits --output "$R/${TOOL}-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  done
done
for L in 384 768 1024; do
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact front8 --qualify --artifact "$ARTIFACT" --length "$L" --rounds 24 --replays 30 --output "$R/qualified-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact "$ARTIFACT" --previous --length "$L" --rounds 64 --replays 40 --output "$R/anthropic-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
