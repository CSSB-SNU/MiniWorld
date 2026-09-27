#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --array=0-3%4
#SBATCH --job-name=trimul-native-width
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_widths_20260922/native-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
widths=(64 256 384 512)
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_widths_20260922/native_bench.py --width "${widths[$SLURM_ARRAY_TASK_ID]}"
