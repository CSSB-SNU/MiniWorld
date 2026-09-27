#!/bin/bash
#SBATCH -J mw-ta-helper-probe
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:08:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/helper-probe-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python "$R/screen_three_score_spill.py"
SOL_VARIANT=baseproducerexp3score timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/baseproducerexp3score-L768.json"
