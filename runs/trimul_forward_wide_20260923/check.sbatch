#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=mw-wide-forward
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_forward_wide_20260923/job-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_forward_wide_20260923/check.py "$@"
