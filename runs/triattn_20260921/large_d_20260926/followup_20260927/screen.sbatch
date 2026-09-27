#!/bin/bash
#SBATCH -J mw-triattn-wide-next-screen
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH --array=0-3%4
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/screen-%A_%a.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
KINDS=(pipe3 share2 n256 k128)
KIND=${KINDS[$SLURM_ARRAY_TASK_ID]}
timeout 180s bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/screen.py" --kind "$KIND" --width 512 --output "$R/screen-${KIND}-${SLURM_ARRAY_JOB_ID}.json"
