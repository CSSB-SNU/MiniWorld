#!/bin/bash
#SBATCH -J mw-training-fwd-candidate-ncu
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/candidate-profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --force-overwrite -o "$R/candidate-profile-${SLURM_JOB_ID}" python -u -B "$R/profile_candidate.py" --artifact "$ARTIFACT" --output "$R/candidate-profile-${SLURM_JOB_ID}.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$R/candidate-profile-${SLURM_JOB_ID}.ncu-rep" --page raw --csv > "$R/candidate-profile-${SLURM_JOB_ID}.csv"
