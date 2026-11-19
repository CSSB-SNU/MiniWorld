#!/bin/bash
#SBATCH --job-name=mw_test_fp32bias
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit/test_fp32bias_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit/test_fp32bias_%j.out
set -uo pipefail
cd /home/psk6950/MiniWorld
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; python scripts/analysis/fp32_inference/test_fp32_pair_bias.py'
echo "==== EXIT $? ===="
