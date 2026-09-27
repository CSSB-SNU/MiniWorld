#!/bin/bash
#SBATCH -J mw-bias-cluster
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/cluster-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
timeout 180s bash "$ENV" python -u -B "$R/check.py" --native-only --output "$R/cluster-correctness-${SLURM_JOB_ID}.json"
for L in 384 768 1024; do
  timeout 180s bash "$ENV" python -u -B "$R/bench_sol.py" --length "$L" --output "$R/cluster-pilot-${SLURM_JOB_ID}-L${L}.json"
done
