#!/bin/bash
#SBATCH --job-name=mw-v120-medium-1b
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/v1.2.0/phase1a/medium_launch_20260917_130505/slurm-13228.log
#SBATCH --open-mode=append
#SBATCH --nodelist=node02
set -euo pipefail
export MW_LAUNCH_DIR=/home/psk6950/MiniWorld/runs/v1.2.0/phase1a/medium_launch_20260917_130505
export MW_ROOT=/home/psk6950/MiniWorld
cd "$MW_LAUNCH_DIR/engine2_resume_20260923"
export PATH="$MW_ROOT/.pixi/envs/cu128/bin:$PATH"
export LD_LIBRARY_PATH="$MW_ROOT/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/engine:$PWD/src:$PWD/libs/team-gm/src:$PWD/scripts"
export MW_RESUME_RUN_SUBDIR=/home/psk6950/MiniWorld/runs/v1.2.0/phase1a/medium_launch_20260917_130505/training/2026-09-17/131543_v1.2.0-phase1a-medium-PF16-L384-triton-4xh100
export MINIWORLD_PWA_TRAIN=1 MINIWORLD_OPM_TRAIN=1
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export MINIWORLD_MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx
export TORCH_EXTENSIONS_DIR=/home/psk6950/MiniWorld/runs/engine2-training-profile-20260923/extension-cache/engine2
export PYTHONPYCACHEPREFIX=/tmp/mw_engine2_resume_${SLURM_JOB_ID}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export TRITON_CACHE_AUTOTUNING=1
export TRITON_CACHE_DIR=/tmp/triton_psk6950_mw_graph_ab
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_psk6950_mw_graph_ab
export MPLCONFIGDIR="$MW_LAUNCH_DIR/matplotlib"
export WANDB_DIR="$MW_LAUNCH_DIR"
export WANDB_ENTITY=team_gm
export WANDB_PROJECT=MiniWorld
export PYTHONUNBUFFERED=1
unset MW_SYNTH_DATA MW_CAPTURE_CACHE MW_MEM_PROFILE MINIWORLD_RUN_AUTOTUNE
trap 'code=$?; date -Is > "$MW_LAUNCH_DIR/ended-at.txt"; echo "$code" > "$MW_LAUNCH_DIR/exit-code.txt"' EXIT
python -u "$MW_LAUNCH_DIR/engine2_resume_20260923/preflight1b.py"
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/run_miniworld_distogram_train.py train \
    --config configs/miniworld/phase1b_distogram_medium_v120.yaml \
    --ckpt "$MW_RESUME_RUN_SUBDIR/checkpoints/last.pt" \
    --job-name v1.2.0-phase1a-medium-PF16-L384-triton-4xh100 \
    train.engine_backend=auto +train.force_trainer=fabric train.compile=true \
    train.grad_accum_steps=32 "train.run_dir=$MW_LAUNCH_DIR/training" \
    "data.train_db.catalog_cache_path=/home/psk6950/MiniWorld/runs/v1.1.0/phase1a/launch_20260916_233206/catalog.arrow"
