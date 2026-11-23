#!/bin/bash
#SBATCH --job-name=mw_x0hat_inf
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/x0hat_inf_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100/x0hat_inf_%j.out
# x0-hat trajectory of the v2.4.0 phase-2a epoch-100 model on FoldBench inputs (8oxk, 9jst), ORIGINAL FoldBench inference settings (EMA, bf16 autocast, engine kernels),
# 1 seed x 1 sample, 200 steps: per step the bond-length deviation of x0-hat vs the CCD ideal lengths and the clash count.
set -uo pipefail
cd /home/psk6950/MiniWorld
OUT=/home/psk6950/MiniWorld/runs/foldbench/x0hat_traj_ep100
CK=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384_from1b/2026-11-18/101134_v240-phase2a-medium-L384-local/checkpoints/epoch=0100.pt
printf '8oxk-assembly1\n9jst-assembly1\n' > "$OUT/index.txt"
export MW_COMPILE_TIMEOUT=1 MINIWORLD_TRANSITION_FORCE_SPLIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${USER}_fb_shared; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 CK="$CK" OUT="$OUT" X0HAT_OUT="$OUT" pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python scripts/analysis/x0hat_bonds/x0hat_inference.py foldbench \
    --config configs/miniworld/phase2a_diffusion_medium_v240.yaml --ckpt "$CK" \
    --index "$OUT/index.txt" --inputs-root /home/psk6950/data/foldbench_v240eval/inputs --output-dir "$OUT/preds" \
    --n-seeds 1 --n-samples 1 --num-recycles 10 --timesteps 200 --msa-subsample --use-ema'
echo "==== EXIT $? ===="
