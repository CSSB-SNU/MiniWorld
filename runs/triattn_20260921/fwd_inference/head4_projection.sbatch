#!/bin/bash
#SBATCH -J mw-head4-ablation
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/head4-projection-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/head4_projection_check.py" --head1 "${HEAD1:-h1kv_stream2}" --head4 "${HEAD4:-h4kv_stream2}" --output "$R/head4-projection-${SLURM_JOB_ID}.json"
