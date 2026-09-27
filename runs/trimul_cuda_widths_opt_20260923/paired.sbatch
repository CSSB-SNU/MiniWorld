#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%4
#SBATCH --job-name=tri-width-check
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_cuda_widths_opt_20260923/paired-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
widths=(64 128 256 384 512)
lengths=(384 768)
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_cuda_widths_opt_20260923/compare_previous.py --width "${widths[$((SLURM_ARRAY_TASK_ID / 2))]}" --length "${lengths[$((SLURM_ARRAY_TASK_ID % 2))]}"
