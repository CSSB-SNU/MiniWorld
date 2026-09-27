#!/bin/bash
#SBATCH -J mw-ta-pipeline
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/cta_pipeline/test-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
for ENDING in 0 1; do
 export ENDING
 for L in 384 768 1024; do
  bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/cta_pipeline/bench_serving.py" --length "$L" --output "$R/cta_pipeline/serving-e${ENDING}-L${L}.json"
 done
done
