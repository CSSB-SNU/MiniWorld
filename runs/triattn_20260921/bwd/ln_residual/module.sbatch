#!/bin/bash
#SBATCH -J mw-ln-module
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/ln_residual/module-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/ln_residual
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_module.py"
for L in 384 768 1024; do
 bash "$ENV" python -u -B "$R/paired.py" --length "$L" --output "$R/paired-${SLURM_JOB_ID}-L${L}.json"
done
for TOOL in memcheck racecheck synccheck; do
 bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=projection_ln_residual_tma|reduce_norm|finish_norm' --error-exitcode 86 python -u -B "$R/check.py" --length 64 --native-only --output "$R/${TOOL}-${SLURM_JOB_ID}.json"
done
