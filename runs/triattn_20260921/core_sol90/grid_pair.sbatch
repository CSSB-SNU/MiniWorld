#!/bin/bash
#SBATCH -J mw-ta-grid-pair
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/grid-pair-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for DIR in 0 1; do
 ENDING="$DIR" GRID_ORDERS=0,3,4 timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench_grid_order.py" --length 768 --output "$R/grid-pair-${SLURM_JOB_ID}-${DIR}.json"
done
