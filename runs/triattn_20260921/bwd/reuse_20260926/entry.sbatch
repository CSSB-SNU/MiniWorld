#!/bin/bash
#SBATCH -J mw-bwd-reuse-entry
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926/entry-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
  timeout 180s bash "$ENV" python -u -B "$R/changed_graph.py" --entry --artifact reuse_qdo --length "$L" --output "$R/entry-${L}-${SLURM_JOB_ID}.json"
done
