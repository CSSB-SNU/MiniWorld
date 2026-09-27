#!/bin/bash
#SBATCH --job-name=swadit
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/swa-%j.log
set -uo pipefail
cd /home/psk6950/MiniWorld
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH="/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
RD=/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927
./.pixi/envs/cu128/bin/python $RD/test_qkvg_cuda.py 2>&1 | grep --line-buffered -vE "^\s*$|Warning"
