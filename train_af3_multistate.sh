#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=324g
#SBATCH -p h100
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 40
#SBATCH -o ./logs/run_af3_ms_v0.1.0.out
#SBATCH -e ./logs/run_af3_ms_v0.1.0.err

  
# export OMP_NUM_THREADS=20
# torchrun --master_port 12242 \
#   --nproc_per_node=4 scripts/run_af3_multistate.py train \
#   --ckpt_dir=checkpoints/ms_v0.1.0/ \
#   --config configs/af3_ms_v0.1.0.yaml \
#   --device=cuda \
#   > ./logs/run_af3_ms_v0.1.0.out 2> ./logs/run_af3_ms_v0.1.0.err \
#   -w

  
export OMP_NUM_THREADS=20
torchrun --master_port 12242 \
  --nproc_per_node=2 scripts/run_af3_multistate.py train \
  --resume_from_ckpt checkpoints/ms_v0.1.0/MiniWorld_epoch=0010.pt \
  --device=cuda \
  > ./logs/run_af3_ms_v0.1.0.out 2> ./logs/run_af3_ms_v0.1.0.err \
  -w