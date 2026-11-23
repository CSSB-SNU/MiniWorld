#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
# ORIGINAL FoldBench inference settings (EMA, default precision, engine fused kernels) + the fp32 pair-bias logits patch (fused_logit_fp32.py) switched by MW_LOGIT_FP32.
# env: INDEX (index file), OUTN (run name under runs/foldbench/fusedlogit/), MW_LOGIT_FP32 (1 = patch on)
set -uo pipefail
cd /home/psk6950/MiniWorld
OUT=/home/psk6950/MiniWorld/runs/foldbench/fusedlogit/${OUTN}
CK=${CKPT:-/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384_from1b/2026-11-18/101134_v240-phase2a-medium-L384-local/checkpoints/epoch=0100.pt}
export MW_COMPILE_TIMEOUT=1 MINIWORLD_TRANSITION_FORCE_SPLIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${USER}_fb_shared; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 CK="$CK" OUT="$OUT" INDEX="$INDEX" MW_LOGIT_FP32="${MW_LOGIT_FP32:-0}" pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python scripts/analysis/fp32_inference/fused_logit_fp32.py foldbench \
    --config configs/miniworld/phase2a_diffusion_medium_v240.yaml --ckpt "$CK" \
    --index "$INDEX" --inputs-root /home/psk6950/data/foldbench_v240eval/inputs --output-dir "$OUT" \
    --n-seeds 5 --n-samples 5 --num-recycles 10 --timesteps 200 --msa-subsample --use-ema'
echo "==== EXIT $? ===="
