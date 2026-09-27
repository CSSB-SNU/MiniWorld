#!/bin/bash
#SBATCH -J mw-triattn-wide-beta-check
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927/accumulate-check-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/large_d_20260926/followup_20260927
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_accumulate.py"
for TOOL in memcheck racecheck synccheck; do
 export CHECK_TOOL=$TOOL
 timeout 240s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 87 python -u -B "$R/check_accumulate.py"
done
