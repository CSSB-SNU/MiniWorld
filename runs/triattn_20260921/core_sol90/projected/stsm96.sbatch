#!/bin/bash
#SBATCH -J mw-ta-projected-stsm96
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected/stsm96-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/build_stsm96.py" > "$R/build-stsm96-${SLURM_JOB_ID}.log" 2>&1
bash "$ENV" python -u "$R/check.py" --variant stsm96 --output "$R/check-stsm96-${SLURM_JOB_ID}.json"
for sanitizer in memcheck racecheck synccheck; do
 bash "$ENV" compute-sanitizer --tool "$sanitizer" --error-exitcode 97 python -u "$R/check.py" --variant stsm96 --length 384 --cases random --output "$R/check-stsm96-${sanitizer}-${SLURM_JOB_ID}.json"
done
