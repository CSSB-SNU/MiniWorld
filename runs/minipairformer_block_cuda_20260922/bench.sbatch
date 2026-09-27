#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=minipair-cuda
#SBATCH --output=/home/psk6950/MiniWorld/runs/minipairformer_block_cuda_20260922/bench-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh env TORCH_EXTENSIONS_DIR=/home/psk6950/MiniWorld/runs/minipairformer_block_cuda_20260922/extensions python -B runs/minipairformer_block_cuda_20260922/bench.py
