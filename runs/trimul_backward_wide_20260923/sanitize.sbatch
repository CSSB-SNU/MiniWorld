#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --job-name=mw-b1-sanitize
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_backward_wide_20260923/sanitize-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_backward_wide_20260923/engine_cache
widths=(256 384)
width=${widths[$SLURM_ARRAY_TASK_ID]}
for tool in memcheck racecheck; do
    bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'regex=mw_wide_b1_prepare|width_b1' --error-exitcode 9 --log-file "runs/trimul_backward_wide_20260923/${tool}-D${width}.log" python -B runs/trimul_backward_wide_20260923/check.py --width "$width" --sanitize
done
