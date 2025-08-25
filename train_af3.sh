#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=80g
#SBATCH -p h100
#SBATCH -w node02
#SBATCH --gres=gpu:h100:4
#SBATCH -c 80
#SBATCH -o ./logs/run_af3_v0.2.2.out
#SBATCH -e ./logs/run_af3_v0.2.2.err

  
export OMP_NUM_THREADS=20
torchrun --master_port 12262 \
  --nproc_per_node=4 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.2.2/ \
  --config configs/af3_triton_v0.2.2.yaml \
  --device=cuda \
  -w

# export OMP_NUM_THREADS=20
# torchrun --master_port 12242 \
#   --nproc_per_node=1 scripts/run_af3.py train \
#   --ckpt_dir=checkpoints/v0.1.6/ \
#   --device=cuda