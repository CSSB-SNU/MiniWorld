#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=mw-d256-ncu
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_20260923/profile-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_20260923/engine_cache
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_d256_bwd_sol90_20260923/collect_ncu.py
