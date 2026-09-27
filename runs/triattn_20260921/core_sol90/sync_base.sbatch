#!/bin/bash
#SBATCH -J mw-ta-sync
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/sync-%j.log
set -euo pipefail
export SOL_SANITIZER=1 SOL_FULLCASE=1 CHECK_PATTERNS=forced_safe
R=/home/psk6950/MiniWorld/runs/triattn_20260921
FILTER="kns=ta_sol_${SOL_VARIANT}"
if [ "$SOL_VARIANT" = installed ]; then FILTER=kns=ta_core_broadcast; fi
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/compute-sanitizer --tool synccheck --kernel-name "$FILTER" --print-limit 2 --error-exitcode 1 python -u -B "$R/core_sol90/check_base.py"
