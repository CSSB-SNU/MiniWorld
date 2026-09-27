#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%3
#SBATCH --job-name=mw-fwd-sanitize
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_forward_wide_20260923/sanitize-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_forward_wide_20260923/engine_cache
widths=(256 256 384 384 512 512)
lengths=(384 768 384 768 384 768)
for tool in memcheck racecheck; do
  bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'kns=mw_wide_' --error-exitcode 9 --log-file "/home/psk6950/MiniWorld/runs/trimul_forward_wide_20260923/${tool}-D${widths[$SLURM_ARRAY_TASK_ID]}-L${lengths[$SLURM_ARRAY_TASK_ID]}.log" python -B runs/trimul_forward_wide_20260923/validate_engine.py --sanitize --width "${widths[$SLURM_ARRAY_TASK_ID]}" --length "${lengths[$SLURM_ARRAY_TASK_ID]}"
done
