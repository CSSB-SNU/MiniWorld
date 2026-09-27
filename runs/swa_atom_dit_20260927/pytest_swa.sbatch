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
#SBATCH --time=01:30:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/swa-%j.log
set -uo pipefail
cd /home/psk6950/MiniWorld
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH="/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
RD=/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927
PY=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin/python
for F in 0 1; do
  echo "===== MINIWORLD_SWA_FUSED=$F  (MiniWorld tests)"
  cd /home/psk6950/MiniWorld && MINIWORLD_SWA_FUSED=$F timeout 2000 $PY -m pytest -q -p no:cacheprovider tests/test_diffusion_bf16.py tests/test_v2_correctness.py tests/test_engine_full_wiring.py 2>&1 | grep --line-buffered -E "passed|failed|FAILED|ERROR" | tail -12
  echo "===== MINIWORLD_SWA_FUSED=$F  (team-gm tests)"
  cd /home/psk6950/MiniWorld/libs/team-gm && MINIWORLD_SWA_FUSED=$F timeout 1500 $PY -m pytest -q -p no:cacheprovider tests/layers/test_swa_atom_attention.py tests/models/test_esmfold2.py 2>&1 | grep --line-buffered -E "passed|failed|FAILED|ERROR" | tail -12
done
