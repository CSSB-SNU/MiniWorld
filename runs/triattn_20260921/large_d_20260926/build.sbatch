#!/bin/bash
#SBATCH -J mw-triattn-wide-build
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/build-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for D in ${HEAD_DIMS:-64 128}; do
  for KIND in fwd dq dkdv; do
    timeout 420s bash "$ENV" python -u -B "$R/native.py" --head-dim "$D" --kind "$KIND"
  done
done
