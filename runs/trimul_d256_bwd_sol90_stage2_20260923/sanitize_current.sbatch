#!/bin/bash
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-1%1
#SBATCH --job-name=mw-d256-stage2-sanitize
#SBATCH --output=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/sanitize-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export OMP_NUM_THREADS=4
export MINIWORLD_ENGINE_JIT_ROOT=/home/psk6950/MiniWorld/runs/trimul_d256_bwd_sol90_stage2_20260923/engine_cache
lengths=(384 768)
export LENGTH=${lengths[$SLURM_ARRAY_TASK_ID]}
if [[ ${FULL_VALIDATE:-0} == 1 ]]; then
 SANITIZE=0 bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_d256_bwd_sol90_stage2_20260923/validate_current.py
fi
export SANITIZE=1
for tool in ${SAN_TOOLS:-memcheck racecheck}; do
 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$tool" --kernel-name 'regex=mw_d256_|GateGradSm90' --error-exitcode 9 --log-file "runs/trimul_d256_bwd_sol90_stage2_20260923/${tool}-${VALIDATION_TAG:-current}-L${LENGTH}.log" python -B runs/trimul_d256_bwd_sol90_stage2_20260923/validate_current.py
done
