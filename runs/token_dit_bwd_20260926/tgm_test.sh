#!/bin/bash
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/miniworld-engine-tdt/src
cd /home/psk6950/MiniWorld/libs/team-gm
python -m pytest -q -p no:cacheprovider -W ignore tests/blocks/test_diffusion_transformer_hoist_sm90.py tests/blocks/test_diffusion_transformer.py 2>&1 | grep -E "^FAILED|^E  |passed|failed|skipped" | tail -15
