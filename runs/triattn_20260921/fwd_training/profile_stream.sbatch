#!/bin/bash
#SBATCH -J mw-stream-ncu
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/stream-profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in baseline ${ARTIFACT:-qg_preload3}; do
    P="$R/stream-profile-${SLURM_JOB_ID}-$A"
    timeout 300s bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum --force-overwrite -o "$P" python -u -B "$R/profile_stream.py" --artifact "$A" --length "${LENGTH:-768}" --output "$P.json"
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$P.ncu-rep" --page raw --csv > "$P.csv"
done
