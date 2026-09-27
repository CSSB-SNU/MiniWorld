#!/bin/bash
#SBATCH -J mw-ta-q-interleave
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/q-interleave-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for half in 0 1; do
 bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-baseqthreehalf${half}.log"
done
for direction in 0 1; do
 ENDING="$direction" timeout --signal=TERM --kill-after=10s 300s bash "$ENV" python -u -B "$R/q_three_interleaved.py" --length "${LENGTH:-768}" --output "$R/q-three-interleaved${RESULT_SUFFIX:-}-${direction}.json"
done
