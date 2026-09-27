#!/bin/bash
#SBATCH -J mw-ta-exp-group
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/exp-group-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for group in 2 8; do
    VARIANT=baseqsmallip${group}
    if bash "$ENV" python -u "$R/build_exp_group.py" "$group" > "$R/build-${VARIANT}.log" 2>&1; then
        if bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"; then
            SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
        else
            echo "RESOURCE SCREEN REJECTED $VARIANT"
        fi
    else
        echo "BUILD FAILED $VARIANT"
    fi
done
