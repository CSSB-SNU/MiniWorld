#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=trimul-offset
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_widths_20260922/offset-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_widths_20260922/sanitize_offset.py
