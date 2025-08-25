#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=45g
#SBATCH -p h100
#SBATCH -w node01
#SBATCH --gres=gpu:h100:4
#SBATCH -c 80
#SBATCH -o ./logs/run_af3_v0.2.0.out
#SBATCH -e ./logs/run_af3_v0.2.0.err

  
export OMP_NUM_THREADS=20
torchrun --master_port 12242 \
  --nproc_per_node=4 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.2.0/ \
  --config configs/af3_triton_v0.2.0.yaml \
  --device=cuda \
  -w

# export OMP_NUM_THREADS=20
# torchrun --master_port 12242 \
#   --nproc_per_node=1 scripts/run_af3.py train \
#   --ckpt_dir=checkpoints/v0.1.6/ \
#   --device=cuda