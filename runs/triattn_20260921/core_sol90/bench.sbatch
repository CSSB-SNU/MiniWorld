#!/bin/bash
#SBATCH -J mw-ta-pipeline
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/test-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
SOL_VARIANT=${SOL_VARIANT:-r6n32} bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/core_sol90/bench.py" --length "${LENGTH:-768}" --output "$R/core_sol90/${SOL_VARIANT:-r6n32}-L${LENGTH:-768}.json"
