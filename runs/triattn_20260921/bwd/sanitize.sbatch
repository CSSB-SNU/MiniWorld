#!/bin/bash
#SBATCH -J mw-triattn-bwd-sanitize
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/sanitize-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for TOOL in memcheck racecheck synccheck; do
  bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 86 python -u -B "$R/check_projection.py" --native-only --output "$R/$TOOL.json"
done
