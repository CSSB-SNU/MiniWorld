#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --job-name=trimul-d128
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_widths_20260922/d128-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
 bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_widths_20260922/native_d128.py --width 128 --no-save-xn
else
 lengths=(384 768)
 bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_widths_20260922/bench.py --width 128 --length "${lengths[$SLURM_ARRAY_TASK_ID]}"
fi
