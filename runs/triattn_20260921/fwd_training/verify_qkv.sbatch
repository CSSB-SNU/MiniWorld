#!/bin/bash
#SBATCH -J mw-qkv-verify
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/qkv-verify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
EXTRA=()
if [[ "${STAGE:-0}" == 1 ]]; then EXTRA+=(--stage); fi
if [[ "${INSTALLED:-0}" == 1 ]]; then EXTRA+=(--installed); fi
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/verify_qkv.py" --artifact "$ARTIFACT" "${EXTRA[@]}" --output "$R/qkv-verify-${SLURM_JOB_ID}.json"
