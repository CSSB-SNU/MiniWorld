#!/bin/bash
#SBATCH -J mw-ta-ref
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step27-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh bash -c \
  "export CUDA_VISIBLE_DEVICES=0; python -u -B $R/probe_ref.py"
