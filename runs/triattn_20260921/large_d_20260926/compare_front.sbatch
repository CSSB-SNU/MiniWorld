#!/bin/bash
#SBATCH -J mw-triattn-wide-front-compare
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/compare-front-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for C in ${WIDTHS:-256 512}; do
  for L in ${LENGTHS:-384 768}; do
    timeout 400s bash "$ENV" python -u -B "$R/compare_front.py" --width "$C" --length "$L" --mode "${MODE:-full}" --output "$R/compare-front-C${C}-L${L}-${MODE:-full}-${SLURM_JOB_ID}.json"
  done
done
