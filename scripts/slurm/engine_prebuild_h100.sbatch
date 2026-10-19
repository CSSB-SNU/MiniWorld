#!/usr/bin/env bash
# Prebuild the engine's CUDA kernels for H100 (sm_90) in this checkout's cu128 environment: every extension and length-bucketed
# cubin the production models use at inference and training shapes, then a no-rebuild check in a new process
# (scripts/prebuild_engine.py). Run it after `pixi install -e cu128` and after every engine pin bump, from the repository root:
#
#   sbatch scripts/slurm/engine_prebuild_h100.sbatch                # --mode all, 8 workers over the 2 GPUs
#   JOBS=2 sbatch scripts/slurm/engine_prebuild_h100.sbatch --mode infer
#
# The log lands next to where you submitted (engine-prebuild-<job id>.log); the cache is the environment's own
# (~/.cache/miniworld_engine_jit/py310-<env hash>, see src/miniworld/__init__.py), so every later job on any H100 node reuses it.
#SBATCH --job-name=engine-prebuild
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=engine-prebuild-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
test -f pyproject.toml || { echo "submit from the MiniWorld repository root" >&2; exit 2; }
export PYTHONUNBUFFERED=1
pixi run --frozen -e cu128 setup-cutlass
pixi run --frozen -e cu128 engine-prebuild --jobs "${JOBS:-8}" "$@"
