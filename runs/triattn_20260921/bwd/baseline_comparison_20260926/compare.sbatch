#!/bin/bash
#SBATCH -J mw-triattn-bwd-compare
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/baseline_comparison_20260926/compare-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/baseline_comparison_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in ${LENGTHS:-384 768 1024}; do
  timeout 540s bash "$ENV" python -u -B "$R/compare.py" --length "$L" --rounds "${BENCH_ROUNDS:-24}" --replays "${BENCH_REPLAYS:-10}" --output "$R/compare-${L}-${SLURM_JOB_ID}.json"
done
