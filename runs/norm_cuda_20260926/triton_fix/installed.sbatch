#!/bin/bash
#SBATCH --job-name=mw-norm-installed
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/norm_cuda_20260926/triton_fix/installed-%j.log
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
fix_root="$PWD/runs/norm_cuda_20260926/triton_fix"
python "$fix_root/installed_smoke.py"
python -m pytest -q .engine-release-2.0.0/tests/numerics/test_layernorm_linear_stats_gpu.py .engine-release-2.0.0/tests/numerics/test_norm_dispatch_cache_gpu.py -k 'graph or empty or dtype_scoped or override_and_off or portable_linear_stats' --disable-warnings
