#!/bin/bash
#SBATCH --job-name=mw_x0hat_train_bf16
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/foldbench/x0hat_train_ep100/x0hat_train_bf16_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/foldbench/x0hat_train_ep100/x0hat_train_bf16_%j.out
# x0-hat bond geometry in the TRAINING setup of the v2.4.0 phase-2a run (epoch-100 raw weights, training data on node02's local NVMe, training noise levels, bf16-mixed).
set -uo pipefail
cd /home/psk6950/MiniWorld
OUT=/home/psk6950/MiniWorld/runs/foldbench/x0hat_train_ep100
CK=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384_from1b/2026-11-18/101134_v240-phase2a-medium-L384-local/checkpoints/epoch=0100.pt
export MW_COMPILE_TIMEOUT=1 MINIWORLD_TRANSITION_FORCE_SPLIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${USER}_fb_shared; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 CK="$CK" OUT="$OUT" pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python scripts/analysis/x0hat_bonds/x0hat_train.py --ckpt "$CK" --items 12 --augment 48 --out "$OUT/summary_raw_bf16mixed.json"'
echo "==== EXIT $? ===="
