#!/bin/bash
#SBATCH -J mw-triattn-bwd-check
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/validate-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_projection.py" --output "$R/correctness.json"
for L in 384 768 1024; do
  bash "$ENV" python -u -B "$R/paired_projection.py" --length "$L" --output "$R/paired-L$L.json"
done
