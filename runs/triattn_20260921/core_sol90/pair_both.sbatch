#!/bin/bash
#SBATCH -J mw-ta-pair-both
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/pair-both-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${SOL_VARIANT:?candidate required}"
LENGTH=${LENGTH:-768}
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${SOL_VARIANT}.log"
for direction in 0 1; do
    ENDING="$direction" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/pair.py" --length "$LENGTH" --output "$R/${SOL_VARIANT}-paired-currentL${LENGTH}-${direction}.json"
done
