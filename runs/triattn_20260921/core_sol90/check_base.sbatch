#!/bin/bash
#SBATCH -J mw-ta-check
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/check-%j.log
set -euo pipefail
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/check_base.py
