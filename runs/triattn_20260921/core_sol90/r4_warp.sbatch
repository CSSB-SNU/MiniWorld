#!/bin/bash
#SBATCH -J mw-ta-r4-warp
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/r4-warp-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=m128r4wp2
if [ "${R4_COOPERATIVE:-0}" = 1 ]; then VARIANT=m128r4coop2; fi
if [ -n "${R4_FENCEFIX:-}" ]; then VARIANT="${VARIANT}f${R4_FENCEFIX}"; fi
bash "$ENV" python -u "$R/build_r4_warp.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_r4_warp.py" "$R/build-${VARIANT}.log"
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
