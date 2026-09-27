#!/bin/bash
#SBATCH -J mw-ta-grid-order
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/grid-order-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/build_grid_order.py" > "$R/build-basegrid.log" 2>&1
bash "$ENV" python "$R/screen_grid_order.py"
timeout --signal=TERM --kill-after=10s 240s bash "$ENV" python -u -B "$R/bench_grid_order.py" --length 768 --output "$R/grid-order-L768.json"
