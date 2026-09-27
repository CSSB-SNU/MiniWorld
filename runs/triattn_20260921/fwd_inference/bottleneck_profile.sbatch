#!/bin/bash
#SBATCH -J mw-triattn-bottleneck
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/bottleneck-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" nvidia-smi --query-gpu=name,memory.total,clocks.current.sm,clocks.max.sm,power.limit --format=csv
for CELL in 384:h4kv_local1 768:hot6t 1024:hot4t; do
  L=${CELL%%:*}; A=${CELL#*:}; P="$R/bottleneck-${A}-${L}-${SLURM_JOB_ID}"
  timeout 300s bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats --section InstructionStats --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum --force-overwrite -o "$P" python -u -B "$R/profile_algorithm.py" --artifact "$A" --length "$L" --output "$P.json"
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$P.ncu-rep" --page raw --csv > "$P.csv"
done
