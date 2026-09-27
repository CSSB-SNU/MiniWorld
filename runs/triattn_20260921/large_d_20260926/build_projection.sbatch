#!/bin/bash
#SBATCH -J mw-triattn-wide-projection
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/projection-build-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
for C in ${WIDTHS:-256 512}; do
 timeout 400s bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/projection.py" --width "$C"
done
