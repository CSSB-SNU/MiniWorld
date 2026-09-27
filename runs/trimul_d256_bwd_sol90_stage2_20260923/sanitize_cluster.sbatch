#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=mw-d256-cluster-sanitize
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/cluster-sanitize-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --version
for tool in memcheck racecheck; do
 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'regex=mw_d256_cluster_' --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-cluster-${SLURM_JOB_ID}.log" python -B "runs/trimul_d256_bwd_sol90_stage2_20260923/${SCRIPT:-check_cluster_smoke.py}"
done
