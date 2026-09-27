#!/bin/bash
#SBATCH -J mw-ta-pipeline
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/cta_pipeline/test-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
for TOOL in racecheck synccheck; do
 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/compute-sanitizer --tool "$TOOL" --kernel-name kns=triattn_tma_pro_pipeline --launch-count 2 --error-exitcode 1 python -u -B "$R/cta_pipeline/check_pro_serving.py"
done
