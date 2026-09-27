#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-1%2
#SBATCH --job-name=mw-d256-verify
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_20260923/verify-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_20260923/engine_cache
lengths=(384 768)
splits=(8 16)
export LENGTH=${lengths[$SLURM_ARRAY_TASK_ID]}
bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_d256_bwd_sol90_20260923/stress.py
for tool in memcheck racecheck; do
 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'kns=mw_d256_b7_' --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_20260923/${tool}-L${LENGTH}.log" python -B runs/trimul_d256_bwd_sol90_20260923/check.py --length "$LENGTH" --splits "${splits[$SLURM_ARRAY_TASK_ID]}" --sanitize
done
