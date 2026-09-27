#!/bin/bash
#SBATCH -J mw-ta-tex-probe
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/texture-exp-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
/usr/local/cuda-12.9/bin/nvcc -O3 --use_fast_math -arch=sm_90a -std=c++17 -Xptxas=-v "$R/texture_exp_probe.cu" -o "$R/texture_exp_probe"
"$R/texture_exp_probe"
