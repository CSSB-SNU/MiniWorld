#!/bin/bash
#SBATCH -J mw-gate-current
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/gate_delta/current-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/gate_delta
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 8 16 64 128; do
 bash "$ENV" python -u -B "$R/check_current.py" --length "$L" --no-bench --output "$R/current-check-${SLURM_JOB_ID}-L${L}.json"
done
for L in 384 768 1024; do
 bash "$ENV" python -u -B "$R/check_current.py" --length "$L" --output "$R/current-pilot-${SLURM_JOB_ID}-L${L}.json"
done
