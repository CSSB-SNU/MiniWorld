#!/bin/bash
#SBATCH -J mw-ta-sanitize
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/sanitize-%j.log
set -euo pipefail
export SOL_SANITIZER=1
R=/home/psk6950/MiniWorld/runs/triattn_20260921
for TOOL in racecheck synccheck memcheck; do
 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/compute-sanitizer --tool "$TOOL" --kernel-name "kns=ta_sol_${SOL_VARIANT}" --error-exitcode 1 python -u -B "$R/core_sol90/check_base.py"
done
