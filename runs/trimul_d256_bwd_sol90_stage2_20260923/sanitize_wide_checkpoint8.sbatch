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
#SBATCH --job-name=mw-wide-checkpoint8
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/wide-checkpoint8-validation-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
task_script=runs/trimul_d256_bwd_sol90_stage2_20260923/validate_wide_checkpoint8.py
SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B "$task_script"
for tool in memcheck racecheck synccheck; do
 filter=()
 if [ "$tool" != memcheck ]; then
  filter=(--kernel-name 'regex=mw_wide384_|mw_wide512_|mw_wide_b7_source|mw_prefix_|mw_wide_output_save_products|mw_wide_bounded_ln|mw_wide_pipe_source|mw_wide_input_ln_reduce|mw_wide_dense_|mw_wide_tile_ln|mw_wide_tma_input|mw_wide_mask_source|mw_wide_tile_norm|mw_wide_front_pre_tma|mw_wide_saved_front_gp|mw_wide_cached_input|mw_wide_cached_ln|mw_wide_four_group_source|mw_wide_saved_gp_pipe|mw_wide_front_pre_transposed|mw_wide_two_group_contract_gp')
 fi
 SANITIZE=1 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" "${filter[@]}" --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-wide-checkpoint8-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" python -B "$task_script"
done
