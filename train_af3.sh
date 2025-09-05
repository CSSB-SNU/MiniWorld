#!/bin/sh 
#SBATCH -J MoE_AF3
#SBATCH --mem=160g
#SBATCH -p h100
#SBATCH -w node01
#SBATCH --gres=gpu:h100:8
#SBATCH -c 160
#SBATCH -o ./logs/run_af3_v0.4.1.out
#SBATCH -e ./logs/run_af3_v0.4.1.err

  
export OMP_NUM_THREADS=20
torchrun --master_port 12272 \
  --nproc_per_node=8 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.4.1/ \
  --config configs/af3_triton_v0.4.1.yaml \
  --device=cuda \
  -w
