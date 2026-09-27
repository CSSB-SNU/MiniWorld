#!/bin/bash
#SBATCH -J mw-ta-persist-screen
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/persistb4screen-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for variant in persistkv2b4; do
    if bash "$ENV" python "$R/build.py" "$variant" > "$R/build-${variant}.log" 2>&1; then
        if bash "$ENV" python "$R/screen_m256_build.py" "$R/build-${variant}.log"; then
            SOL_VARIANT="$variant" bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${variant}-L768.json"
        else
            echo "REGISTER/ASYNC SCREEN REJECTED $variant"
        fi
    else
        echo "BUILD FAILED $variant; see build log"
    fi
done
