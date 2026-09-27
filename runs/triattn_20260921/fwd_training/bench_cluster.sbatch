#!/bin/bash
#SBATCH -J mw-fwd-cluster-bench
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/bench-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
# L64 build pilots do not wrap the two-stage multicast barriers.
timeout 120s bash "$ENV" python -u -B "$R/check.py" --artifact "$ARTIFACT" --baseline-artifact cooperative_q1_s2 --length 256 --output "$R/cluster-wrap-${SLURM_JOB_ID}.json"
for L in 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check.py" --artifact "$ARTIFACT" --baseline-artifact cooperative_q1_s2 --length "$L" --output "$R/core-${SLURM_JOB_ID}-L${L}.json"
    timeout 240s bash "$ENV" python -u -B "$R/module_check.py" --artifact "$ARTIFACT" --baseline-artifact cooperative_q1_s2 --length "$L" --mode bench --output "$R/bench-${SLURM_JOB_ID}-L${L}.json"
done
