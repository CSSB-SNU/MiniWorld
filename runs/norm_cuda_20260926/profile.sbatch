#!/bin/bash
#SBATCH --job-name=mw-norm-ncu
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/norm_cuda_20260926/profile-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export PATH="$PWD/.pixi/envs/cu128/bin:/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=4 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
export PYTHONPYCACHEPREFIX="$PWD/runs/norm_cuda_20260926/pycache-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/.engine-release-2.0.0/src"
export TORCH_EXTENSIONS_DIR="$PWD/runs/norm_cuda_20260926/build-0"
ncu --metrics 'dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,gpu__time_duration.sum' --kernel-name 'regex:forward_norm|forward_vec|backward_fused|affine_finish' --launch-skip 9 --launch-count 3 python runs/norm_cuda_20260926/ncu_probe.py
