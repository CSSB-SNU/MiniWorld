#!/bin/bash
#SBATCH -J mw-ta-debug
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/debug-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
SOL_VARIANT=${SOL_VARIANT:-split4} bash "$R/../anthropic_adoption_20260919/env.sh" python -u -B "$R/core_sol90/debug_split.py" --length 768 --output /tmp/debugsplit.json
