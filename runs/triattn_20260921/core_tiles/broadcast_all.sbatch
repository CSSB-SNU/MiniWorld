#!/bin/bash
#SBATCH -J mw-ta-pipeline
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_tiles/test-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
for ending in 0 1; do
 for length in 768 1024; do
  CORE_VARIANT=broadcast ENDING=$ending bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/core_tiles/bench.py" --length "$length" --output "$R/core_tiles/broadcast-L${length}-e${ending}.json"
 done
done
