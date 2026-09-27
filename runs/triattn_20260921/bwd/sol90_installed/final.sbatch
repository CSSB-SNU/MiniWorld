#!/bin/bash
#SBATCH -J mw-bwd-verify-final
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/sol90_installed/final-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/sol90_installed
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_guards.py"
CONTEXT_FROM_PACKAGE=1 bash "$ENV" python -u -B "$R/check_cold_compile.py"
for L in 384 768 1024; do
 bash "$ENV" python -u -B "$R/paired.py" --length "$L" --output "$R/paired-${SLURM_JOB_ID}-L${L}.json"
done
bash "$ENV" python -u -B "$R/check_amp.py"
