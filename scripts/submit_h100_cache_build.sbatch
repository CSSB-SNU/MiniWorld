#!/bin/bash
#SBATCH --job-name=mw-h100-cache
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:7
#SBATCH --cpus-per-task=112
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/h100_build_20260915_retry1/build-%j.log

set -euo pipefail
cd /home/psk6950/MiniWorld
export BUILD_RUN_DIR="$PWD/runs/h100_build_20260915_retry1"
export PATH="$PWD/.pixi/envs/cu128/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export MINIWORLD_MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx
export MINIWORLD_PLAN_CACHE_DIR="$BUILD_RUN_DIR/plans"
export TRITON_CACHE_DIR=/tmp/triton_cache_psk6950_cgaudit
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_psk6950_cgaudit
export PYTHONPATH="$BUILD_RUN_DIR/package:$PWD/libs/team-gm/src"
export MAX_JOBS=12
export PYTHONUNBUFFERED=1

finish() {
    result=$?
    python - "$BUILD_RUN_DIR" "$result" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "job_exit.json"
path.write_text(json.dumps({"exit_code": int(sys.argv[2]),
                           "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
PY
}
trap finish EXIT

python - <<'PY'
import datetime, json, os, pathlib, torch
import miniworld_engine
from miniworld_engine.autotune.native import source_identity
run = pathlib.Path(os.environ["BUILD_RUN_DIR"])
metadata = json.loads((run / "metadata.json").read_text())
assert torch.cuda.device_count() == 7, "expected seven allocated GPUs"
assert source_identity() == metadata["native_source_identity"], "snapshot source changed"
status = {"job_id": os.environ["SLURM_JOB_ID"], "state": "running",
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "package": miniworld_engine.__file__, "source_identity": source_identity(),
          "gpus": [torch.cuda.get_device_name(i) for i in range(7)]}
(run / "job_started.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
PY

# Default incremental tuning and resume retain previously valid measurements.
# The CLI verifies/derives its plan, merges successful shards, and checks coverage.
set +e
python -m miniworld_engine.cli build all --gpus all \
    --shards "$BUILD_RUN_DIR/shards" --compile-jobs 12 \
    --units-per-gpu 1 --unit-timeout-seconds 7200 --keep-triton-cache
build_result=$?
python scripts/publish_engine_build_cache.py "$BUILD_RUN_DIR" --build-exit "$build_result"
publish_result=$?
if [ "$build_result" -ne 0 ]; then
    exit "$build_result"
fi
exit "$publish_result"
