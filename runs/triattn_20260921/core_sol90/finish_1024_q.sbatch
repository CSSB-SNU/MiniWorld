#!/bin/bash
#SBATCH -J mw-ta-finish-q1024
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/finish-q1024-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/install_1024_q.py"
VERIFY_PREFIX=serving-q1024 VERIFY_FLAG1024=536870912 bash "$R/verify_fast.sbatch"
PROFILE_NAME=installed-q1024-L768 PROFILE_LENGTH=768 bash "$R/profile_installed.sbatch"
PROFILE_NAME=installed-q1024-L1024 PROFILE_LENGTH=1024 bash "$R/profile_installed.sbatch"
