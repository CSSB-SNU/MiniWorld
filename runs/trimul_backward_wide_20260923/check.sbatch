#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --job-name=mw-b1-wide
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_backward_wide_20260923/check-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_backward_wide_20260923/engine_cache
widths=(256 384 512)
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_backward_wide_20260923/check.py --width "${widths[$SLURM_ARRAY_TASK_ID]}" --length "${LENGTH:-384}"
