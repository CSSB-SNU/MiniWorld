#!/bin/bash
#SBATCH -J mw-ta-deep-bias
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/deep-bias-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for slots in 4 6; do
    VARIANT=basebias${slots}${BIAS_GUARD:-unroll}
    if bash "$ENV" python -u "$R/build_deep_bias.py" "$slots" "${BIAS_GUARD:-unroll}" > "$R/build-${VARIANT}.log" 2>&1; then
        if bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"; then
            SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
        else
            echo "RESOURCE SCREEN REJECTED $VARIANT"
        fi
    else
        echo "BUILD FAILED $VARIANT"
    fi
done
