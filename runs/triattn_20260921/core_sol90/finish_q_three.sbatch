#!/bin/bash
#SBATCH -J mw-ta-finish-qthree
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/finish-qthree-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/install_q_three.py"
VERIFY_PREFIX=serving-qthree bash "$R/verify_fast.sbatch"
PROFILE_NAME=installed-qthree bash "$R/profile_installed.sbatch"
