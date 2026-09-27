#!/bin/bash
#SBATCH -J mw-triattn-wide-proj-tiles
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/projection-tiles-%j.log
set -euo pipefail
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/tune_projection.py
