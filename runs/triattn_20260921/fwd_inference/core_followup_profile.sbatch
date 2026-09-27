#!/bin/bash
#SBATCH -J mw-triattn-coreprofile
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/core-followup-profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${ARTIFACT:?}"
for A in hot4t "$ARTIFACT"; do
  P="$R/coreprofile-${A}-1024-${SLURM_JOB_ID}"
  timeout 300s bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats --section InstructionStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --force-overwrite -o "$P" python -u -B "$R/profile_algorithm.py" --artifact "$A" --length 1024 --output "$P.json"
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$P.ncu-rep" --page raw --csv > "$P.csv"
done
