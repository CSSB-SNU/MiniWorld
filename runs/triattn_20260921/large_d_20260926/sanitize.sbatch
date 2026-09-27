#!/bin/bash
#SBATCH -J mw-triattn-wide-sanitize
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --array=0-5%2
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/sanitize-%A_%a.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
TOOLS=(memcheck racecheck synccheck)
D=$((64 << (SLURM_ARRAY_TASK_ID / 3)))
TOOL=${TOOLS[$((SLURM_ARRAY_TASK_ID % 3))]}
timeout 1500s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 87 python -u -B "$R/check_native.py" --native-only --head-dim "$D" --length 256 --mask mixed --output "$R/sanitize-h${D}-${TOOL}-${SLURM_ARRAY_JOB_ID}.json"
