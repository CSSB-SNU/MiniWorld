#!/bin/bash
#SBATCH -J mw-triattn-wide-fast-check
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/fast-check-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
timeout 120s bash "$ENV" python -u -B "$R/check_fast.py"
for TOOL in memcheck racecheck synccheck; do
 export CHECK_TOOL=$TOOL
 timeout 240s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 87 python -u -B "$R/check_fast.py"
done
for C in 256 512; do
 timeout 250s bash "$ENV" python -u -B "$R/check_selected_graph.py" --width "$C" --output "$R/selected-graph-C${C}-${SLURM_JOB_ID}.json"
done
