#!/bin/bash
#SBATCH --job-name=mw-cute-activate
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/native_tuning_dev/activation-%j.log
set -euo pipefail
cd /home/psk6950/MiniWorld
export PATH="$PWD/.pixi/envs/cu128/bin:$PATH"
export PYTHONPATH="$PWD/libs/team-gm/src"
python - <<'PY'
import json,pathlib
root=pathlib.Path.cwd()
passed=json.loads((root/'runs/native_tuning_dev/final_passed.json').read_text())
manifest=json.loads((root/'patches/native-tuning-manifest.json').read_text())
assert passed['passed'] and passed['native_identity']==manifest['after_identity']
PY
python scripts/activate_native_tuning.py \
  --package "$PWD/.pixi/envs/cu128/lib/python3.10/site-packages/miniworld_engine" \
  --after-build "$PWD/runs/h100_build_20260915_retry1" \
  --report "$PWD/runs/native_tuning_dev/activation.json"
