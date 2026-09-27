#!/bin/bash
#SBATCH -J mw-q-fusion-promote
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/q-promotion-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/promote_q_fusion.py" --artifact q_only_head4 --qualify 17762 --verify 17765 --bench 17756 --sanitize 17766 --profile 17767
