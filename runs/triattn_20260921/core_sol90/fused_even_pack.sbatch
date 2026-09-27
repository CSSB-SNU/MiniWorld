#!/bin/bash
#SBATCH -J mw-ta-even-pack
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/even-pack-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
MODE=${MODE:-partial}
VARIANT=baseevenpack${MODE}v2
bash "$ENV" python -u "$R/build_fused_even_pack.py" "$MODE" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
