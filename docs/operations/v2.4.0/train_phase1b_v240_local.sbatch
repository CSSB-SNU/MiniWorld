#!/bin/bash
#SBATCH --job-name=mw_v240_p1b_med
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=8-00:00:00
#SBATCH --output=/home/psk6950/MiniWorld-scratch/phase1b-v240/train_%j.out
#SBATCH --error=/home/psk6950/MiniWorld-scratch/phase1b-v240/train_%j.out
# MiniWorld v2.4.0 phase 1b (L768 / 8192 atoms), started from the phase-1a epoch-600 full state, in a NEW run directory
# (--ckpt + train.resume_run_subdir = the new directory: the trainer loads weights / optimizer / scheduler / epoch from the checkpoint and
# writes into the given directory; without resume_run_subdir it would continue the phase-1a run directory). A crash restarts from this run's last.pt.
# Resources: the first run asked 140 CPUs / 800 GB and used ~14 cores on average (TotalCPU 7d11h over 12.5 h; data_wait ~30 ms of a 20 s step), so 40 CPUs / 256 GB / 8 loader workers per rank.
set -uo pipefail
mkdir -p /home/psk6950/MiniWorld-scratch/phase1b-v240
cd /home/psk6950/MiniWorld
ROOT=/home/psk6950/MiniWorld/runs/v2.4.0/phase1b/medium_distogram_L768
RUN=$ROOT/run1
SRC=/home/psk6950/MiniWorld/runs/v2.4.0/phase1a/medium_distogram_L384_scratch/2026-11-01/234255_5040bd46/checkpoints/epoch=0600.pt
echo "=== job $SLURM_JOB_ID node $SLURM_JOB_NODELIST  commit $(git rev-parse --short HEAD)  run $RUN  from $SRC"
before=""
for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt  $(date)"
  if [ "$attempt" = 1 ] && [ ! -f "$RUN/checkpoints/last.pt" ]; then
    CKPT="$SRC"; NEWRUN=1
  else
    CKPT="$RUN/checkpoints/last.pt"; NEWRUN=1
  fi
  echo "ckpt $CKPT"
  CONDA_OVERRIDE_CUDA=12.8 CKPT="$CKPT" RUN="$RUN" ROOT="$ROOT" pixi run --frozen -e cu128 bash -c '
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 MW_COMPILE_TIMEOUT=1
    export PYTHONFAULTHANDLER=1
    export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld WANDB__SERVICE_WAIT=300
    export MW_SAMPLE_TIMING_DIR=/home/psk6950/MiniWorld-scratch/phase1b-v240/sample_timing_${SLURM_JOB_ID}; mkdir -p "$MW_SAMPLE_TIMING_DIR"
    export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_v240"; mkdir -p "$TRITON_CACHE_DIR"
    export PYTHONPATH=/home/psk6950/MiniWorld/libs/FoldForge/src:/home/psk6950/MiniWorld/scripts:${PYTHONPATH:-}
    export MW_WANDB_NEW_RUN=1
    torchrun --standalone --nproc_per_node=4 scripts/run_miniworld_distogram_train.py train \
      --config configs/miniworld/phase1b_distogram_medium_v240_local.yaml \
      --job-name v240-phase1b-medium-L768-local --ckpt "$CKPT" +train.resume_run_subdir="$RUN" \
      train.run_dir="$ROOT" train.use_wandb=true train.num_workers=8 train.prefetch_factor=4
  '
  rc=$?
  echo "==== attempt $attempt EXIT $rc ===="
  [ $rc -eq 0 ] && break
  now=$(stat -c %Y "$RUN/checkpoints/last.pt" 2>/dev/null || echo 0)
  [ "$now" = "$before" ] && { echo "no new checkpoint since the last attempt: stopping"; break; }
  before=$now
done
