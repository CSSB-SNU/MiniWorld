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
PYTHONNOUSERSITE=1 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --kernel-name-base demangled --kernel-name regex:triattn_tma_pro_pipeline --launch-count 1 --section Occupancy --metrics launch__registers_per_thread,launch__shared_mem_per_block -o "$R/cta_pipeline/pro-occupancy" --force-overwrite python -u -B "$R/cta_pipeline/check_pro_serving.py"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/cta_pipeline/pro-occupancy.ncu-rep" --page raw --csv > "$R/cta_pipeline/pro-occupancy.csv"
