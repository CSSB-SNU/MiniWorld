#!/bin/sh 
#SBATCH -J MoE_AF3
#SBATCH --mem=160g
#SBATCH -p h100
#SBATCH -w node01
#SBATCH --gres=gpu:h100:4
#SBATCH -c 80
#SBATCH -o ./logs/run_af3_v0.4.3.out
#SBATCH -e ./logs/run_af3_v0.4.3.err

  
export OMP_NUM_THREADS=20
torchrun --master_port 12043 \
  --nproc_per_node=4 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.4.3/ \
  --config configs/af3_triton_v0.4.3.yaml \
  --device=cuda \
  -w
