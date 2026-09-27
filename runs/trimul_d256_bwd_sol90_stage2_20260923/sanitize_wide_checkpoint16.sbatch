#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%1
#SBATCH --job-name=mw-wide-checkpoint16
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/wide-checkpoint16-validation-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
task_script=runs/trimul_d256_bwd_sol90_stage2_20260923/validate_wide_checkpoint16.py
SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B "$task_script"
for tool in memcheck racecheck synccheck; do
 filter=()
 if [ "$tool" != memcheck ]; then
  filter=(--kernel-name 'regex=mw_wide_staged_epilogue_gp')
 fi
 SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" "${filter[@]}" --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-wide-checkpoint16-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
done

# Validate the new LN consumer under both inherited D512 fallback paths.
if [ "$SLURM_ARRAY_TASK_ID" -ge 2 ]; then
for branch in FORCE_NORM_OVERFLOW FORCE_ROW_OVERFLOW; do
 for tool in memcheck racecheck synccheck; do
  env "$branch=1" SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'regex=mw_wide_stats_ln|mw_wide_lowreg_delta_norm|mw_wide_apply_delta_norm|mw_wide_compact_|mw_wide_gather_changed_norm|mw_wide_scatter_changed_proj' --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-wide-checkpoint16-${branch}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
 done
done

fi
