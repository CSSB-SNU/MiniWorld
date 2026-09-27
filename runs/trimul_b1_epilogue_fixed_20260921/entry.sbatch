#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --job-name=tri-restore
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_b1_epilogue_fixed_20260921/entry-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_b1_epilogue_fixed_20260921/verify_entry.py
