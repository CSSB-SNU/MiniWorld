#!/bin/bash
#SBATCH -J mw-ta-numerics
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/scaled-q-model-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
SOL_VARIANT=baseearlyfence bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/scaled_q_model.py" --length 768 --output "$R/scaled_q_model.json"
