#!/bin/bash
#SBATCH --job-name=synth_speed
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/MiniWorld-scratch/synth-speed/synth_%x_%j.out
# Real train path (CUDA-graph trainer, Adam, EMA, grad accum) with +train.synth_data=true: ONE static synthetic batch at the max bucket shape, so no data loading at all =
# the GPU-only step time of the phase-1a nr_nt setups. 1 GPU, grad_accum GA micro-steps per optimizer step; ms/sample = step time / GA.
# env: PF (8 or 16), GA (default 16), EPOCHS (default 4), STEPS (optimizer steps per epoch, default 12)
set -uo pipefail
cd /home/psk6950/MiniWorld
: "${PF:?set PF=8 or 16}"
GA=${GA:-16}; EPOCHS=${EPOCHS:-4}; STEPS=${STEPS:-12}
ITEMS=$((GA * STEPS))
RUN=/home/psk6950/MiniWorld-scratch/synth-speed/run_pf${PF}_${SLURM_JOB_ID}
mkdir -p "$RUN"
echo "=== synth speed PF=$PF GA=$GA epochs=$EPOCHS steps/epoch=$STEPS node $SLURM_JOB_NODELIST commit $(git rev-parse --short HEAD)"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_v240"; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 RUN="$RUN" PF=$PF GA=$GA EPOCHS=$EPOCHS ITEMS=$ITEMS pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_DEBUG=WARN MW_COMPILE_TIMEOUT=1 PYTHONFAULTHANDLER=1 PYTHONPATH=/home/psk6950/MiniWorld/scripts:${PYTHONPATH:-}
  torchrun --standalone --nproc_per_node=1 scripts/run_miniworld_distogram_train.py train \
    --config configs/miniworld/phase1a_distogram_medium_v240_nr_nt_pf${PF}_chainfix.yaml --job-name synth-speed-pf${PF} \
    train.run_dir="$RUN" train.use_wandb=false +train.synth_data=true train.num_workers=2 train.prefetch_factor=2 \
    train.num_epoch=$EPOCHS train.train_item=$ITEMS train.grad_accum_steps=$GA train.save_freq=1000'
echo "==== EXIT $? ===="
