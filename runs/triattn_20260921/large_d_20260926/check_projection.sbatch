#!/bin/bash
#SBATCH -J mw-triattn-wide-proj-check
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/projection-check-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for C in 256 512; do
 timeout 180s bash "$ENV" python -u -B "$R/check_projection.py" --width "$C" --output "$R/projection-check-C${C}-${SLURM_JOB_ID}.json"
 for TOOL in memcheck racecheck synccheck; do
  timeout 180s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 87 python -u -B "$R/check_projection.py" --native-only --width "$C" --output "$R/projection-${TOOL}-C${C}-${SLURM_JOB_ID}.json"
 done
done
