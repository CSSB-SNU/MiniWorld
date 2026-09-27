#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --job-name=b1-sol-calibrate
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_b1_sol90_20260921/calibrate-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_b1_sol90_20260921/calibrate.py
