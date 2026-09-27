#!/bin/bash
#SBATCH -J mw-below80-installed
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80/final-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_integrity.py"
bash "$ENV" python -u -B "$R/../sol90_installed/check_guards.py"
CONTEXT_FROM_PACKAGE=1 bash "$ENV" python -u -B "$R/../sol90_installed/check_cold_compile.py"
unset BIAS_ARTIFACT DQ_ARTIFACT LN_ARTIFACT
for L in 384 768 1024; do
    bash "$ENV" python -u -B "$R/paired_combined.py" --length "$L" --output "$R/combined-${SLURM_JOB_ID}-L${L}.json"
done
bash "$ENV" python -u -B "$R/../sol90_installed/check_amp.py"
