#!/bin/bash
#SBATCH -J mw-triattn-wide-next-build
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH --array=0-5%6
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/build-%A_%a.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
KINDS=(pipe3 share2 n256 k128 bias16 bias16 accumulate accumulate_fast)
WIDTHS=(512 512 512 512 256 512 512 512)
timeout 600s bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/build.py" --kind "${KINDS[$SLURM_ARRAY_TASK_ID]}" --width "${WIDTHS[$SLURM_ARRAY_TASK_ID]}"
