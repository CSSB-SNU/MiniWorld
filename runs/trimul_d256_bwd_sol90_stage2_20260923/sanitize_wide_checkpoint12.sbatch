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
#SBATCH --job-name=mw-wide-checkpoint12
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/wide-checkpoint12-validation-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
task_script=runs/trimul_d256_bwd_sol90_stage2_20260923/validate_wide_checkpoint12.py
SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B "$task_script"
# All inherited D384 kernels are qualified in checkpoint10 (18048/17970).
# Race/sync instrument the only new kernel; memcheck covers the complete path.
for tool in memcheck racecheck synccheck; do
 filter=()
 if [ "$tool" != memcheck ]; then
  filter=(--kernel-name 'regex=mw_wide_cached_forward_norm')
 fi
 SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" "${filter[@]}" --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-wide-checkpoint12-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
done
