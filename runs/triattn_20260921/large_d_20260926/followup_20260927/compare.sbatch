#!/bin/bash
#SBATCH -J mw-triattn-wide-next-compare
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH --array=0-3%4
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/compare-%A_%a.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
WIDTHS=(256 256 512 512)
LENGTHS=(384 768 384 768)
C=${WIDTHS[$SLURM_ARRAY_TASK_ID]}
L=${LENGTHS[$SLURM_ARRAY_TASK_ID]}
PROJ=${PROJECTION:-auto}
if [ "$PROJ" = auto ]; then
 if [ "$C" = 256 ]; then PROJ=prior; else PROJ=accumulate; fi
fi
timeout 900s bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh python -u -B "$R/compare.py" --width "$C" --length "$L" --projection "$PROJ" --compact "${COMPACT:-1}" --output "$R/compare-C${C}-L${L}-${PROJ}-b${COMPACT:-1}-${SLURM_ARRAY_JOB_ID}.json"
