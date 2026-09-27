#!/bin/bash
#SBATCH -J mw-qkv-phase
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/stream-phase-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/native.py" --artifact qkv_compact_phase
timeout 180s bash "$ENV" python -u -B "$R/qkv_phase_probe.py" --output "$R/stream-phase-${SLURM_JOB_ID}.json"
