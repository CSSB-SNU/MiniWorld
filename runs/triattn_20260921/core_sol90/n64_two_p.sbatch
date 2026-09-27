#!/bin/bash
#SBATCH -J mw-ta-n64-two-p
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/n64-two-p-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=m128s2p2qrn64
bash "$ENV" python -u "$R/build_n64_two_p.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_m256_build.py" "$R/build-${VARIANT}.log"
if rg -q C7520 "$R/build-${VARIANT}.log"; then exit 2; fi
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
