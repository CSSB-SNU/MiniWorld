#!/bin/bash
#SBATCH -J mw-stream-promote
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/qkv-promote-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/promote_qkv_fusion.py" --artifact "$ARTIFACT" \
    --qualify "$QUALIFY" --verify "$VERIFY" --bench "$BENCH" \
    --sanitize "$SANITIZE" --sanitize1024 "$SANITIZE1024" --profile "$PROFILE"
