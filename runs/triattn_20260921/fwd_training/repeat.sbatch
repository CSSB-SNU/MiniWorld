#!/bin/bash
#SBATCH -J mw-training-fwd-repeat
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/repeat-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
    bash "$ENV" python -u -B "$R/check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/core-${SLURM_JOB_ID}-L${L}.json"
    bash "$ENV" python -u -B "$R/module_check.py" --artifact "$ARTIFACT" --length "$L" --mode bench --output "$R/bench-${SLURM_JOB_ID}-L${L}.json" --baseline-artifact cooperative_q2 --rounds 32 --replays 40
done
