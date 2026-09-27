#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=3
#SBATCH --job-name=mw-wide-checkpoint13
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/wide-checkpoint13-validation-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
task_script=runs/trimul_d256_bwd_sol90_stage2_20260923/validate_wide_checkpoint13.py
SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B "$task_script"
# Inherited D512 kernels and norm/projection overflow paths are qualified in checkpoint11 (18108).
# Race/sync instrument the only new kernel; memcheck covers the complete path.
for tool in memcheck racecheck synccheck; do
 filter=()
 if [ "$tool" != memcheck ]; then
  filter=(--kernel-name 'regex=mw_wide_affine_output_ln')
 fi
 SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" "${filter[@]}" --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-wide-checkpoint13-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
done
