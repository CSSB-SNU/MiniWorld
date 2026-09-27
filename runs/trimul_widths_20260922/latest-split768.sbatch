#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=1-1
#SBATCH --job-name=trimul-latest
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_widths_20260922/latest-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
lengths=(384 768)
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_widths_20260922/latest_d128.py --length "${lengths[$SLURM_ARRAY_TASK_ID]}" --split-b7
