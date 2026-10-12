#!/bin/bash
#SBATCH --job-name=mw-fixed-compile
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/phase1_compile_graph_fix/compiled-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/runs/phase1_compile_graph_fix/package:$PWD/src:$PWD/libs/team-gm/src:$PWD/scripts"
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/tmp/triton_psk6950_mw_graph_ab
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_psk6950_mw_graph_ab
export TRITON_CACHE_AUTOTUNING=1
for rec in 1 4; do
  for graph in off manual; do
    .pixi/envs/cu128/bin/python -u scripts/benchmark_miniworld_training_graph.py --compile --catalog-snapshot --graph "$graph" --recycles "$rec" --batch runs/miniworld_graph_ab/real_batches.pt --output "runs/phase1_compile_graph_fix/r${rec}-${graph}-compile.json"
  done
done
