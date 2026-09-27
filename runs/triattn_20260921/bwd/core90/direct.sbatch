#!/bin/bash
#SBATCH -J mw-below80
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90/direct-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
    bash "$ENV" python -u -B "$R/compare_native.py" --kind "$KIND" --artifact "$ARTIFACT" --length "$L" --output "$R/direct-${SLURM_JOB_ID}-L${L}.json"
done
