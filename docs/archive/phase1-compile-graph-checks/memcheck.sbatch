#!/bin/bash
#SBATCH --job-name=mw-fa4-bwd-compile
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/phase1_compile_graph_fix/memcheck-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/runs/phase1_compile_graph_fix/package:$PWD/src:$PWD/libs/team-gm/src:$PWD/scripts"
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/tmp/triton_psk6950_mw_graph_ab
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_psk6950_mw_graph_ab
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 42 .pixi/envs/cu128/bin/python -m pytest tests/test_phase1_flash_backward_compile.py -x -q --junitxml=runs/phase1_compile_graph_fix/memcheck.xml
