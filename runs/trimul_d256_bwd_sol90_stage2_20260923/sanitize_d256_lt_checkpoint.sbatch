#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%1
#SBATCH --job-name=mw-d256-lt-checkpoint
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/d256-lt-checkpoint-validation-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
task_script=runs/trimul_d256_bwd_sol90_stage2_20260923/validate_d256_lt_checkpoint.py
SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B "$task_script"
for tool in memcheck racecheck synccheck; do
 filter=()
 if [ "$tool" != memcheck ]; then
  filter=(--kernel-name 'regex=mw_d256_|mw_wide_dense_|mw_wide_cached_input|mw_prefix_transpose|GateGradSm90')
 fi
 SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" "${filter[@]}" --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-d256-lt-checkpoint-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
done
