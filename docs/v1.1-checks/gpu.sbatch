#!/bin/bash
#SBATCH --job-name=mw-v110-distogram
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/v110_distogram_checks/gpu-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src:$PWD/libs/team-gm/src"
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_psk6950_v110_loss
export TRITON_CACHE_DIR=/tmp/triton_psk6950_v110_loss
export OMP_NUM_THREADS=4
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 42 \
  .pixi/envs/cu128/bin/python -m pytest tests/test_distogram_v110.py -k cuda -q \
  --junitxml=runs/v110_distogram_checks/gpu.xml
