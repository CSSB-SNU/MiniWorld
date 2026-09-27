#!/bin/bash
#SBATCH -J mw-triattn-outprofile
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/outproj-profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768 1024; do
  for DIR in starting ending; do
    FLAGS=(); if [[ "$DIR" == ending ]]; then FLAGS+=(--ending); fi
    P="$R/outprofile-${ARTIFACT:-outproj_c1}-${L}-${DIR}-${SLURM_JOB_ID}"
    timeout 180s bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section LaunchStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --force-overwrite -o "$P" python -u -B "$R/profile_outproj.py" --artifact "${ARTIFACT:-outproj_c1}" --length "$L" "${FLAGS[@]}" --output "$P.json"
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$P.ncu-rep" --page raw --csv > "$P.csv"
  done
done
