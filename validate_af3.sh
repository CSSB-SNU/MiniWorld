#!/bin/sh 
#SBATCH -J MoE_AF3
#SBATCH --mem=80g
#SBATCH -p h100
#SBATCH -w node01
#SBATCH --gres=gpu:h100:1
#SBATCH -c 20
#SBATCH -o ./logs/run_af3_v0.4.1_valid.out
#SBATCH -e ./logs/run_af3_v0.4.1_valid.err

export OMP_NUM_THREADS=20
torchrun --master_port 11021 scripts/full_validate.py validate \
  --ckpt checkpoints/v0.4.1/MiniWorld_epoch=1000.pt \
  --config configs/af3_triton_v0.4.1_validate.yaml \
  --device=cuda