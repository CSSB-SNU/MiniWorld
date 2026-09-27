#!/bin/bash
#SBATCH -J mw-ta-half-exp-rate
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/half-exp-rate-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90a --ptxas-options=-v "$R/half_exp_bench.cu" -o "$R/half_exp_bench" > "$R/build-half-exp-rate-${SLURM_JOB_ID}.log" 2>&1
"$R/half_exp_bench"
