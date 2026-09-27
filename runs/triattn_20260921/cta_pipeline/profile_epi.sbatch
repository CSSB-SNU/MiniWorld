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
PYTHONNOUSERSITE=1 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors.sum -o "$R/cta_pipeline/epi-profile" --force-overwrite python -u -B "$R/cta_pipeline/roof_epi.py" --length 768 --output "$R/cta_pipeline/roof.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/cta_pipeline/epi-profile.ncu-rep" --page raw --csv > "$R/cta_pipeline/epi-profile.csv"
