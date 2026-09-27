#!/bin/bash
#SBATCH -J mw-training-fwd-full-sanitize
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/full-sanitize-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for TOOL in memcheck racecheck synccheck; do
    bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=training_fwd' --error-exitcode 86 python -u -B "$R/check.py" --artifact "${ARTIFACT:-cooperative_q2}" --length 768 --native-only --cases mixed all_masked --output "$R/${TOOL}-${SLURM_JOB_ID}-L768.json"
done
