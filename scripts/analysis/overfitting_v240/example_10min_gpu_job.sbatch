#!/bin/bash
#SBATCH -J v120_ablate_msa
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o /home/psk6950/MiniWorld-scratch/fb-v240/ablate_msa_%j.out
#SBATCH -e /home/psk6950/MiniWorld-scratch/fb-v240/ablate_msa_%j.out
# v1.2.0 (ep100/200/300/400/900) on 20 training-DB antibody-antigen items: full MSA vs the antibody chains' MSA removed.
set -uo pipefail
cd /home/psk6950/MiniWorld
R=/home/psk6950/MiniWorld/runs/v1.2.0
A=$R/phase1a/medium_launch_20260917_130505/training/2026-09-17/131543_v1.2.0-phase1a-medium-PF16-L384-triton-4xh100/checkpoints
B=$R/phase1b/main_20260930/training/2026-10-02/090256_v1.2.0-phase1a-medium-PF16-L384-triton-4xh100/checkpoints
F=/home/psk6950/MiniWorld-scratch/fb-v240
M=phase1a_distogram_medium_v120
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen -e cu128 bash -c "
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH\" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MW_COMPILE_TIMEOUT=1
  export TRITON_CACHE_DIR=/tmp/triton_cache_\${USER}_v240; mkdir -p \$TRITON_CACHE_DIR
  python $F/eval_ablate_msa.py --repo /home/psk6950/MiniWorld --config phase1a_distogram_medium_v240_local --n 20 --out $F/ablate_msa \
    ep100=$A/epoch=0100.pt@$M ep200=$A/epoch=0200.pt@$M ep300=$A/epoch=0300.pt@$M ep400=$A/epoch=0400.pt@$M ep900=$B/epoch=0900.pt@$M"
echo "==== EXIT \$? ===="
