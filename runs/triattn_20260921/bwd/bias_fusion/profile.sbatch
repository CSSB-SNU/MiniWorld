#!/bin/bash
#SBATCH -J mw-bias-profile
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion/profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/bias_fusion
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for ROWS in ${FUSION_PROFILE_ROWS:-0 4 8}; do
  bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors.sum --force-overwrite -o "$R/profile-${SLURM_JOB_ID}-r${ROWS}" python -u -B "$R/roof.py" --rows "$ROWS" --length "${FUSION_PROFILE_LENGTH:-768}"
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/profile-${SLURM_JOB_ID}-r${ROWS}.ncu-rep" --page raw --csv > "$R/profile-${SLURM_JOB_ID}-r${ROWS}.csv"
done
