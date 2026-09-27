#!/bin/bash
#SBATCH -J k1k3-claude
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH -q cssb_h100
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/claude_logs/hold-%j.out
nvidia-smi --query-gpu=index,name,memory.used --format=csv
sleep infinity
