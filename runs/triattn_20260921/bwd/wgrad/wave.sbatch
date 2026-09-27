#!/bin/bash
#SBATCH -J mw-wgrad
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/wgrad/wave-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/wgrad
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for SPEC in '384 2240 2304 4096' '768 4480 4544 8960' '1024 8000 8192 15936'; do
 read -r -a VALUES <<< "$SPEC"
 L="${VALUES[0]}"
 bash "$ENV" python -u -B "$R/check.py" --length "$L" --splits "${VALUES[@]:1}" --output "$R/wave-${SLURM_JOB_ID}-L${L}.json"
done
