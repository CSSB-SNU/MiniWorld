#!/bin/bash
#SBATCH -J mw-ta-ptxas-graph
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:12:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/ptxas-graph-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
PREFIX=${CUBIN_OUT_PREFIX:-ptxas-graph}
for ending in 0 1; do
 ENDING="$ending" timeout --signal=TERM --kill-after=10s 240s bash "$ENV" python -u -B "$R/bench_ptxas_graph.py" --length 768 --output "$R/${PREFIX}-e${ending}.json"
done
