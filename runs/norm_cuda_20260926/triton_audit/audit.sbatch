#!/bin/bash
#SBATCH --job-name=mw-triton-audit
#SBATCH --array=0-2
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/norm_cuda_20260926/triton_audit/job-%A_%a.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export PATH="$PWD/.pixi/envs/cu128/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=4 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
export PYTHONPYCACHEPREFIX="$PWD/runs/norm_cuda_20260926/pycache-audit-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/.engine-release-2.0.0/src"
export TORCH_EXTENSIONS_DIR="$PWD/runs/norm_cuda_20260926/opt/build-0"
audit_root="$PWD/runs/norm_cuda_20260926/triton_audit"
python -u "$audit_root/audit.py" --group "$SLURM_ARRAY_TASK_ID"
if [ "$SLURM_ARRAY_TASK_ID" = 2 ]; then
  audit_cases="0 1 2"
else
  audit_cases="0 1 2 3 4"
fi
for audit_case in $audit_cases; do
  ncu --profile-from-start off --cache-control none --csv --log-file "$audit_root/ncu-${SLURM_ARRAY_TASK_ID}-${audit_case}.csv" --metrics 'dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active' python "$audit_root/audit.py" --group "$SLURM_ARRAY_TASK_ID" --case "$audit_case" --profile
done
