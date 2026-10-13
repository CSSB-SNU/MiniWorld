#!/usr/bin/env bash
# The test suite on a B200 node with the environment it needs to be green.
#   gpuq run -g 6 -n tests -- bash scripts/b200/run_tests.sh [pytest args...]
# Three things the bare `pytest` run got wrong on this cluster:
#   * nvcc picked the conda gcc 14 headers, which it cannot parse -> use the system g++ as the host compiler
#   * /tmp is mounted noexec, so inductor and Triton could not map their shared objects -> caches under $W
#   * the JIT extensions were built without pixi_env.sh
set -uo pipefail
W=${W:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950}
source "$W/pixi_env.sh" >/dev/null 2>&1
cd "$(dirname "$0")/../.."
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"
export PYTHONPATH=$PWD/src:${ENGINE_SRC:-$W/miniworld-engine/src} PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_EXTENSIONS_DIR=$W/.cache/torch_extensions_mwtests
export TRITON_CACHE_DIR=$W/scratch/cache/triton_tests TORCHINDUCTOR_CACHE_DIR=$W/scratch/cache/inductor_tests TMPDIR=$W/scratch/tmp
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
exec "$W/mwenv/.pixi/envs/default/bin/python3" -m pytest -q -p no:cacheprovider --tb=short -rf "${@:-tests}"
