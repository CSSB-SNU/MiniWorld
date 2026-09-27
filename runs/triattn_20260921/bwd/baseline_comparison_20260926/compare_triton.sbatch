#!/bin/bash
#SBATCH -J mw-original-triton-compare
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:35:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/baseline_comparison_20260926/triton-compare-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/baseline_comparison_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in ${LENGTHS:-384 768 1024}; do
  timeout 600s bash "$ENV" python -u -B "$R/${BENCH_SCRIPT:-compare_triton.py}" --length "$L" --rounds "${BENCH_ROUNDS:-24}" --replays "${BENCH_REPLAYS:-10}" --output "$R/triton-compare-${L}-${SLURM_JOB_ID}.json"
done
