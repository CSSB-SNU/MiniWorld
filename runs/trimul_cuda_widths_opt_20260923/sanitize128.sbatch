#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=tri-width-check
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_cuda_widths_opt_20260923/sanitize128-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
for tool in memcheck racecheck; do
 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --error-exitcode 99 --target-processes all --log-file "runs/trimul_cuda_widths_opt_20260923/$tool-D128-L768.log" python -B runs/trimul_cuda_widths_opt_20260923/sanitize128.py
done
