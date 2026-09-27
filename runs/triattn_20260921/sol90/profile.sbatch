#!/bin/bash
#SBATCH -J mw-ta-sol90
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/sol90/profile-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" /usr/local/cuda/bin/ncu --target-processes all -k 'regex:triatt_prologue|triatt_epilogue|triattn_m1_kernel' -s 3 -c 3 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --metrics dram__bytes_read.sum,dram__bytes_write.sum -o "$R/sol90/current" --force-overwrite python -u -B "$R/sol90/profile_kernels.py" --length 768 --output "$R/sol90/profile.json"
/usr/local/cuda/bin/ncu --import "$R/sol90/current.ncu-rep" --page raw --csv > "$R/sol90/current.csv"
