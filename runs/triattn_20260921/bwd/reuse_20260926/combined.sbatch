#!/bin/bash
#SBATCH -J mw-bwd-reuse-full
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926/combined-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export BASELINE_FROM_PACKAGE=1
export BENCH_ROUNDS=${BENCH_ROUNDS:-24}
export BENCH_REPLAYS=${BENCH_REPLAYS:-20}
for L in ${LENGTHS:-384 768 1024}; do
  timeout 420s bash "$ENV" python -u -B "$R/combined.py" --length "$L" --output "$R/combined-${L}-${SLURM_JOB_ID}.json"
done
