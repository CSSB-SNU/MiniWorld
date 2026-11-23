#!/bin/bash
#SBATCH --job-name=mw_dfp32_9jst
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-ep100-dfp32-9jst/fp32_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-ep100-dfp32-9jst/fp32_%j.out
# lowest-lDDT FoldBench target of the v2.4.0 2a epoch-100 run (9jst-assembly1) predicted with the trunk in bf16 and ONLY the diffusion module in fp32 (train.precision=native: no autocast; model.diffusion.dtype=fp32).
set -uo pipefail
cd /home/psk6950/MiniWorld
OUT=/home/psk6950/MiniWorld/runs/foldbench/v2.4.0-2a-ep100-dfp32-9jst
CK=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384_from1b/2026-11-18/101134_v240-phase2a-medium-L384-local/checkpoints/epoch=0100.pt
export MW_COMPILE_TIMEOUT=1 MINIWORLD_TRANSITION_FORCE_SPLIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${USER}_fb_shared; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 CK="$CK" OUT="$OUT" pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python scripts/analysis/fp32_inference/run_dfp32_foldbench.py foldbench \
    --config configs/miniworld/phase2a_diffusion_medium_v240.yaml --ckpt "$CK" \
    --index scripts/analysis/fp32_inference/index_9jst.txt \
    --inputs-root /home/psk6950/data/foldbench_v240eval/inputs --output-dir "$OUT" \
    --n-seeds 5 --n-samples 5 --num-recycles 10 --timesteps 200 --msa-subsample --use-ema \
    train.precision=native model.diffusion.dtype=fp32'
echo "==== EXIT $? ===="
