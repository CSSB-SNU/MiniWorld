#!/bin/bash
#SBATCH -J mw-ta-half2-model
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/half2-exp-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/build_half2_exp_probe.py" > "$R/build-half2-exp-probe.log" 2>&1
for direction in 0 1; do
 SOL_VARIANT=baseqsmallcodeo2 ENDING="$direction" timeout --signal=TERM --kill-after=10s 240s bash "$ENV" python -u -B "$R/half2_exp_model.py" --length 768 --output "$R/half2-exp-model-${direction}.json"
done
