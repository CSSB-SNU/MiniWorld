#!/bin/bash
#SBATCH -J mw-wgrad
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/wgrad/qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/wgrad
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
WGRAD_ARTIFACT=shared_z_tensor bash "$ENV" python -u -B "$R/check.py" --length 64 --splits 2304 8960 15936 --output "$R/fp64-${SLURM_JOB_ID}.json"
bash "$ENV" python -u -B "$R/check_module.py"
for TOOL in memcheck racecheck synccheck; do
 WGRAD_ARTIFACT=shared_z_tensor bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=grouped_wgrad|reduce_wgrad' --error-exitcode 86 python -u -B "$R/check.py" --length 64 --split 512 --native-only --output "$R/${TOOL}-${SLURM_JOB_ID}.json"
done
