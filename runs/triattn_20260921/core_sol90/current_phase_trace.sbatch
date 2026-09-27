#!/bin/bash
#SBATCH -J mw-ta-phase-trace
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:06:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/current-phase-trace-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export CUBIN_CASES="$R/current-phase-trace-cases.json"
for ending in 0 1; do
    ENDING="$ending" timeout --signal=TERM --kill-after=5s 150s bash "$ENV" python -u -B "$R/bench_current_phase_trace.py" --length 768 --output "$R/current-phase-trace-e${ending}.json"
done
