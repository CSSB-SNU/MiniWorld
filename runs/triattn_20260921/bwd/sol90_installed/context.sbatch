#!/bin/bash
#SBATCH -J mw-gate-context
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/sol90_installed/context-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/sol90_installed
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export PYTHONFAULTHANDLER=1
bash "$ENV" python -u -B "$R/check_context.py"
bash "$ENV" python -u -B "$R/check_cold_compile.py"
