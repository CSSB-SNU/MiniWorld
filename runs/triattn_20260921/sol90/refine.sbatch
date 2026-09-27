#!/bin/bash
#SBATCH -J mw-ta-gluon
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/sol90/refine-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/sol90/bench_pro_refine.py" --length 768 --output "$R/sol90/pro_refine.json"
