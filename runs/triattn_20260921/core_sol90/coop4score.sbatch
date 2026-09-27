#!/bin/bash
#SBATCH -J mw-ta-coop-four
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/coop-four-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=coop3s4ss
if [ "${COOP_QREGS:-0}" = 1 ]; then VARIANT=coop3s4qr; fi
if [ "${COOP_OPAQUE:-0}" = 1 ]; then VARIANT="${VARIANT}z"; fi
if [ "${COOP_UNROLL:-0}" = 1 ]; then VARIANT="${VARIANT}full"; fi
bash "$ENV" python -u "$R/build_coop4score.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_coop4score.py" "$R/build-${VARIANT}.log"
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
