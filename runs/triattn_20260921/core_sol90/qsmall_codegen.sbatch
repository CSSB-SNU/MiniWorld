#!/bin/bash
#SBATCH -J mw-ta-codegen
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/codegen-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for MODE in o2 ru0; do
    VARIANT=baseqsmallcode${MODE}
    bash "$ENV" python -u "$R/build_qsmall_codegen.py" "$MODE" > "$R/build-${VARIANT}.log" 2>&1
    if bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"; then
        /usr/local/cuda-12.9/bin/cuobjdump -sass "$R/build/triattn_sol_${VARIANT}/m1_1073741824.cuda.o" > "$R/${VARIANT}-hot.sass"
        SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
    fi
done
