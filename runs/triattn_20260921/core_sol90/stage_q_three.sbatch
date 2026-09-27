#!/bin/bash
#SBATCH -J mw-ta-stage-qthree
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/stage-qthree-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u "$R/stage_q_three.py" > "$R/build-qthree-package.log" 2>&1
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-qthree-package.log"
export SOL_VARIANT=qthreepackaged
SOL_FULLCASE=1 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=0 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=1 CHECK_SCALE=0.25 CHECK_PATTERNS=forced_safe bash "$ENV" python -u -B "$R/check_base.py"
for tool in racecheck synccheck memcheck; do
 SOL_FULLCASE=1 SOL_SANITIZER=1 bash "$ENV" /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --kernel-name kns=ta_core_broadcast --error-exitcode 1 python -u -B "$R/check_base.py"
done
