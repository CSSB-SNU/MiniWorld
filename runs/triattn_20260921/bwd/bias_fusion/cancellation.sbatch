#!/bin/bash
#SBATCH -J mw-bias-cancel
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:05:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/cancellation-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/check_cancellation.py"
