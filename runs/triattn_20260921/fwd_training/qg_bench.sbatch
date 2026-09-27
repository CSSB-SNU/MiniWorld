#!/bin/bash
#SBATCH -J mw-qg-bench
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/qg-bench-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/qg_check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/qg-core-${SLURM_JOB_ID}-L${L}.json"
done
