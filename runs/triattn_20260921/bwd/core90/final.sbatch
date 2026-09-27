#!/bin/bash
#SBATCH -J mw-below80-installed
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90/final-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
unset BIAS_ARTIFACT DQ_ARTIFACT LN_ARTIFACT BASELINE_FROM_PACKAGE BASELINE_CHECKPOINT BENCH_ROUNDS BENCH_REPLAYS
export FROM_PACKAGE=1
bash "$ENV" python -u -B "$R/check_integrity.py"
bash "$ENV" python -u -B "$R/../priority_pass/check_wgrad.py"
bash "$ENV" python -u -B "$R/../sol90_installed/check_guards.py"
CONTEXT_FROM_PACKAGE=1 bash "$ENV" python -u -B "$R/../sol90_installed/check_cold_compile.py"
for L in 384 768 1024; do
    bash "$ENV" python -u -B "$R/combined.py" --length "$L" --output "$R/combined-${SLURM_JOB_ID}-L${L}.json"
done
bash "$ENV" python -u -B "$R/../sol90_installed/check_amp.py"
