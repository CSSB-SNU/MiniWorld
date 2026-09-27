#!/bin/bash
#SBATCH -J mw-below80-full
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/priority_pass/combined-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/priority_pass
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
    bash "$ENV" python -u -B "$R/combined.py" --length "$L" --output "$R/combined-${SLURM_JOB_ID}-L${L}.json"
done
