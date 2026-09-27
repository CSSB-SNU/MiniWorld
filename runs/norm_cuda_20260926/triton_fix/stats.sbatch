#!/bin/bash
#SBATCH --job-name=mw-norm-stats-check
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/norm_cuda_20260926/triton_fix/stats-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export PATH="$PWD/.pixi/envs/cu128/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=4 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
export PYTHONPYCACHEPREFIX="$PWD/runs/norm_cuda_20260926/pycache-audit-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/runs/norm_cuda_20260926/triton_fix/candidate"
export TORCH_EXTENSIONS_DIR="$PWD/runs/norm_cuda_20260926/opt/build-0"
fix_root="$PWD/runs/norm_cuda_20260926/triton_fix"
python -m pytest -q "$fix_root/test_linear.py" --disable-warnings -k saved_stats -x
