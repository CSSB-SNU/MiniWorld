#!/bin/bash
#SBATCH -J mw-triattn-qfull768
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:05:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/qfull768-compare-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/finalist_compare.py" --base hot6t --artifact qfull4v2 --length 768 --output "$R/finalist-hot6t-qfull4v2-768-${SLURM_JOB_ID}.json"
