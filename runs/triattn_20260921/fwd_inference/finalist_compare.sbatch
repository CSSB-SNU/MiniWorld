#!/bin/bash
#SBATCH -J mw-triattn-finalists
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/finalist-compare-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in n40_qhalf4 qfull4v2; do
  bash "$ENV" python -u -B "$R/finalist_compare.py" --base qhalf4 --artifact "$A" --output "$R/finalist-qhalf4-${A}-${SLURM_JOB_ID}.json"
done
bash "$ENV" python -u -B "$R/n40_pair_check.py" --artifact qfull4v2 --base hot4t --length 1024 --output "$R/native-pair-qfull4v2-1024-${SLURM_JOB_ID}.json"
