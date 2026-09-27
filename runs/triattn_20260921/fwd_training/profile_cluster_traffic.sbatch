#!/bin/bash
#SBATCH -J mw-fwd-cluster-traffic
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/traffic-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for A in cooperative_q1_s2 cooperative_cluster2_v2 cooperative_cluster2_early; do
    P="$R/traffic-${SLURM_JOB_ID}-${A}"
    timeout 120s bash "$ENV" /usr/local/cuda/bin/ncu --profile-from-start off --clock-control none --cache-control none --metrics lts__t_sectors.sum,lts__d_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --force-overwrite -o "$P" python -u -B "$R/profile_candidate.py" --artifact "$A" --output "$P.json"
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda/bin/ncu --import "$P.ncu-rep" --page raw --csv > "$P.csv"
done
