#!/bin/bash
#SBATCH -J mw-ta-projected-ncu
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected/profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/projected
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
PYTHONNOUSERSITE=1 bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --kernel-name-base demangled --kernel-name 'regex:triattn_projected_fused_.*attention<.*0>' --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors.sum -o "$R/${PROJECTED_VARIANT}-profile" --force-overwrite python -u -B "$R/projected_roof.py" --length 768 --output "$R/profile-context.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/${PROJECTED_VARIANT}-profile.ncu-rep" --page raw --csv > "$R/${PROJECTED_VARIANT}-profile.csv"
