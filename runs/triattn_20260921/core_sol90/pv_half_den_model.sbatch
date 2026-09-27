#!/bin/bash
#SBATCH -J mw-ta-pvhalf-model
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/pv-half-den-model-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/pv_half_den_model.py" --length 768 --output "$R/pv-half-den-model.json"
