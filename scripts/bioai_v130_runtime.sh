#!/usr/bin/env bash
# Source from BioAI batch jobs: process-local runtime, including validated D64 fix.
set -euo pipefail
v130_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
v130_runtime=${MINIWORLD_V130_RUNTIME:-"$v130_root/runs/v1.3.0/implementation_20260924/runtime"}
v130_env="$v130_root/.pixi/envs/cu128"
if [[ ! -f "$v130_runtime/engine-src/miniworld_engine/integrations/optimizer.py" ]]; then
    echo "Missing v1.3 Engine 2 runtime: $v130_runtime" >&2
    return 1
fi
module load cuda/12.8
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH="$v130_env/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$v130_runtime/python:$v130_runtime/engine-src:$v130_root/src:$v130_root/libs/team-gm/src:$v130_root/scripts"
export LD_LIBRARY_PATH="$v130_env/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_ENTITY=team_gm
