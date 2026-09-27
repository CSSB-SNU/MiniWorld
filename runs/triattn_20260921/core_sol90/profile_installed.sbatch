#!/bin/bash
#SBATCH -J mw-ta-pipeline
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/installed-profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
PROFILE_NAME=${PROFILE_NAME:-installed-q1024-L768}
PROFILE_LENGTH=${PROFILE_LENGTH:-768}
PYTHONNOUSERSITE=1 bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh /usr/local/cuda/bin/ncu --profile-from-start off --kernel-name-base demangled --kernel-name regex:triattn_m1_kernel --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors.sum -o "$R/core_sol90/${PROFILE_NAME}-profile" --force-overwrite python -u -B "$R/core_pipeline/roof_core.py" --length "$PROFILE_LENGTH" --output "$R/core_sol90/${PROFILE_NAME}-roof.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/core_sol90/${PROFILE_NAME}-profile.ncu-rep" --page raw --csv > "$R/core_sol90/${PROFILE_NAME}-profile.csv"
