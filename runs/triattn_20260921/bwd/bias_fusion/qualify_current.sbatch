#!/bin/bash
#SBATCH -J mw-bias-qualified
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/qualify-current-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check.py" --output "$R/correctness-${SLURM_JOB_ID}.json"
for L in ${FUSION_LENGTHS:-384 768 1024}; do
  bash "$ENV" python -u -B "$R/paired_current.py" --length "$L" --output "$R/paired-current-${SLURM_JOB_ID}-L${L}.json"
done
bash "$ENV" python -u -B "$R/check_api.py"
for TOOL in memcheck racecheck synccheck; do
  bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=grouped_dkdv|reduce_bias' --error-exitcode 86 python -u -B "$R/check.py" --native-only --output "$R/${TOOL}-${SLURM_JOB_ID}.json"
done
