#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --array=0-1%2
#SBATCH --job-name=b7-nextrow-full
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_b7_nextrow_20260921/full-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
lengths=(384 768)
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_b7_nextrow_20260921/bench.py --length "${lengths[$SLURM_ARRAY_TASK_ID]}" "$@"
