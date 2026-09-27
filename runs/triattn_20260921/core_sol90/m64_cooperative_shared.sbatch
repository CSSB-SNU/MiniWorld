#!/bin/bash
#SBATCH -J mw-ta-m64-coop-p
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/m64-coop-p-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=m64coop2s4p3qr
ARGS=()
if [[ "${DEEP_KV:-0}" == 1 ]]; then VARIANT+=kv4b2; ARGS+=(--deep-kv); fi
bash "$ENV" python -u "$R/build_m64_cooperative_shared.py" "${ARGS[@]}" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_cooperative_shared_probability.py" "$R/build-${VARIANT}.log"
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
