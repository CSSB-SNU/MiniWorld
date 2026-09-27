#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=mw-pool-profile
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/profile-wide23-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
report="runs/trimul_d256_bwd_sol90_stage2_20260923/profile-wide23-short-${SLURM_JOB_ID}"
metrics='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'
bash runs/anthropic_adoption_20260919/env.sh ncu --profile-from-start off --target-processes all --metrics "$metrics" --force-overwrite -o "$report" python -B runs/trimul_d256_bwd_sol90_stage2_20260923/profile_wide23_short.py
bash runs/anthropic_adoption_20260919/env.sh ncu --import "${report}.ncu-rep" --csv --page raw > "${report}.csv"
