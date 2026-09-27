#!/bin/bash
#SBATCH -J mw-ta-serving
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/serving-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
VERIFY_PREFIX=${VERIFY_PREFIX:-serving-q1024}
for direction in 0 1; do
 for length in 768 1024; do
  ENDING=$direction bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/core_sol90/verify_fast_serving.py" --length "$length" --output "$R/core_sol90/${VERIFY_PREFIX}-L${length}-e${direction}.json"
 done
done
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/test_pkg.py"
