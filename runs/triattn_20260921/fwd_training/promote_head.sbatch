#!/bin/bash
#SBATCH -J mw-fwd-head-promote
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/promote-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
# Independent64-round confirmation17568 passed every FWD95% lower bound > 1
# and every F+B median > 1; repeat the same criterion after actual installation.
bash "$ENV" python -u -B "$R/promote_incremental.py" --artifact cooperative_head2 --baseline-artifact cooperative_q1_s2 --baseline-promotion 17345 --qualify 17564 --bench 17568 --profile 17566 --sanitize 17565 --min-forward-speedup 1.0 --require-forward-ci --installed-rounds 64
