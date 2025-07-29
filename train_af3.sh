#!/bin/sh 
#SBATCH -J run_af3
#SBATCH --mem=120g
#SBATCH -p h100
#SBATCH -w node01
#SBATCH --gres=gpu:h100:4
#SBATCH -c 80
#SBATCH -o ./logs/run_af3_v0.1.6.out
#SBATCH -e ./logs/run_af3_v0.1.6.err

# python scripts/run_af3.py train \
#   --config configs/af3_triton.yaml \
#   --ckpt_dir=checkpoints/ \
#   --device=cuda \
#   -w 

# export OMP_NUM_THREADS=20
# torchrun --master_port 12345 \
#   --nproc_per_node=4 scripts/run_af3.py train \
#   --config configs/af3_triton_v0.0.0.yaml \
#   --ckpt_dir=checkpoints/hybrid_norm/ \
#   --device=cuda \
#   -w

# export OMP_NUM_THREADS=20
# torchrun --master_port 12435 \
#   --nproc_per_node=2 scripts/run_af3.py train \
#   --config configs/af3_triton_v0.1.0.yaml \
#   --ckpt_dir=checkpoints/v0.1.0/ \
#   --device=cuda \
#   -w


# export OMP_NUM_THREADS=20
# torchrun --master_port 12495 \
#   --nproc_per_node=1 scripts/run_af3.py train \
#   --resume_from_ckpt MiniWorld_grad_explod.pt \
#   --ckpt_dir=checkpoints/v0.1.0_debug/ \
#   --device=cuda \
  # -w

# export OMP_NUM_THREADS=20
# torchrun --master_port 12425 \
#   --nproc_per_node=2 scripts/run_af3.py train \
#   --config configs/af3_triton_v0.1.2.yaml \
#   --ckpt_dir=checkpoints/v0.1.2/ \
#   --device=cuda \
#   -w  

# export OMP_NUM_THREADS=20
# torchrun --master_port 12425 \
#   --nproc_per_node=4 scripts/run_af3.py train \
#   --ckpt_dir=checkpoints/v0.1.0_debug_resume/ \
#   --resume_from_ckpt checkpoints/v0.1.0/MiniWorld_epoch=0103.pt \
#   --device=cuda \
#   -w
  # --resume_from_ckpt checkpoints/v0.1.0/MiniWorld_epoch=0108.pt \
  # --ckpt_dir=checkpoints/v0.1.0_debug/ \



# export OMP_NUM_THREADS=20
# torchrun --master_port 12425 \
#   --nproc_per_node=1 scripts/run_af3.py train \
#   --config configs/af3_triton_v0.1.1.yaml \
#   --ckpt_dir=checkpoints/v0.1.1/ \
#   --device=cuda \
#   -w  



# export OMP_NUM_THREADS=20
# torchrun --master_port 12425 \
#   --nproc_per_node=4 scripts/run_af3.py train \
#   --ckpt_dir=checkpoints/v0.1.3/ \
#   --config configs/af3_triton_v0.1.3.yaml \
#   --device=cuda \
#   -w

  
export OMP_NUM_THREADS=20
torchrun --master_port 12224 \
  --nproc_per_node=4 scripts/run_af3.py train \
  --ckpt_dir=checkpoints/v0.1.6/ \
  --config configs/af3_triton_v0.1.6.yaml \
  --device=cuda \
  -w