#!/bin/bash
#SBATCH -J mw-ta-half-bias-model
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/half-bias-model-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for DIRECTION in 0 1; do
 ENDING=$DIRECTION SOL_VARIANT=baseqsmallcodeo2 bash "$ENV" python -u -B "$R/half_bias_transport_model.py" --length 768 --output "$R/half-bias-model-${DIRECTION}.json"
done
