#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=45g
#SBATCH -p gpu
#SBATCH -w gpu03
#SBATCH --gres=gpu:A6000:1
#SBATCH -c 16
#SBATCH -o ./logs/run_af3_v0.1.6.out
#SBATCH -e ./logs/run_af3_v0.1.6.err

  
export OMP_NUM_THREADS=20
torchrun --master_port 12242 \
  --nproc_per_node=1 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.1.6/ \
  --config configs/af3_triton_v0.1.6.yaml \
  --device=cuda \
  # -w