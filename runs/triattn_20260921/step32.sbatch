#!/bin/bash
#SBATCH -J mw-ta-codex
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step32-%j.out
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
hostname
for L in 768 384 1024; do
 ENDING=0 bash "$ENV" python -u -B "$R/verify_order.py" --length "$L" --output "$R/logs/step32-L$L.json"
done
ENDING=1 bash "$ENV" python -u -B "$R/verify_order.py" --length 768 --output "$R/logs/step32-ending-L768.json"
