#!/bin/bash
#SBATCH -J mw-ta-small-ones
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/small-ones-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/build_small_ones.py" > "$R/build-basef40smallones.log" 2>&1
bash "$ENV" python "$R/screen_f40_build.py" "$R/build-basef40smallones.log"
SOL_VARIANT=basef40smallones bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/basef40smallones-L768.json"
