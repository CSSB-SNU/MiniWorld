#!/bin/bash
#SBATCH -J mw-ta-twocta-s1
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/twocta-s1-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=m128r2s1p1twocta
if [ "${TWOCTA_PERIOD:-16}" != 16 ]; then VARIANT="${VARIANT}p${TWOCTA_PERIOD}"; fi
if [ "${TWOCTA_FENCE:-0}" = 1 ]; then VARIANT="${VARIANT}f"; fi
if [ "${TWOCTA_SCALAR:-0}" = 1 ]; then VARIANT="${VARIANT}sc"; fi
if [ "${TWOCTA_DENMMA:-0}" = 1 ]; then VARIANT="${VARIANT}mma"; fi
bash "$ENV" python -u "$R/build_two_cta_single_score.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_two_cta_single_score.py" "$R/build-${VARIANT}.log"
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
