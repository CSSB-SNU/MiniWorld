#!/bin/bash
#SBATCH -J mw-ta-n16-latency
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/n16-warp-latency-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${N16_VARIANT:?Set the qualified candidate name}"
bash "$ENV" python "$R/screen_n16_warp_latency.py" "$N16_VARIANT"
SOL_VARIANT="$N16_VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${N16_VARIANT}-L768.json"
