#!/bin/bash
#SBATCH -J mw-ta-factor-model
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/factor-model-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for direction in 0 1; do
    SOL_VARIANT=baseqsmallcodeo2 ENDING="$direction" bash "$ENV" python -u -B "$R/factored_bias_exp_model.py" --length 768 --output "$R/factored-bias-exp-model-${direction}.json"
done
