#!/bin/bash
#SBATCH --job-name=mw_v240_p2a_med
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --nodelist=node02
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=140
#SBATCH --mem=800G
#SBATCH --time=8-00:00:00
#SBATCH --output=/home/psk6950/MiniWorld-scratch/phase2a-v240/train_%j.out
#SBATCH --error=/home/psk6950/MiniWorld-scratch/phase2a-v240/train_%j.out
# MiniWorld v2.4.0 phase 2a (NOT SUBMITTED YET): the frozen v2.4.0 phase-1a trunk (epoch-600 EMA seed) + a new EDM diffusion head, medium, L384,
# node02 local NVMe, epochs chained, W&B on forge team_gm/MiniWorld as a new run. A crash restarts from the newest last.pt of this root (up to 5 times).
# This is the training run itself, so the long time limit of the phase-1a run is kept (the 10-minute rule is for verification jobs).
set -uo pipefail
cd /home/psk6950/MiniWorld
ROOT=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/medium_diffusion_L384
SEED=/home/psk6950/MiniWorld/runs/v2.4.0/phase2a/seed_ema600/epoch=0600_ema.pt
[ -f "$SEED" ] || { echo "seed $SEED is missing: run make_seed_v240.py on epoch=0600.pt first"; exit 2; }
echo "=== job $SLURM_JOB_ID node $SLURM_JOB_NODELIST  commit $(git rev-parse --short HEAD)  root $ROOT  seed $SEED"
before=""
for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt  $(date)"
  if [ "$attempt" = 1 ]; then
    CK="--ckpt $SEED"
  else
    LAST=$(ls -t "$ROOT"/*/*/checkpoints/last.pt 2>/dev/null | head -1)
    [ -n "$LAST" ] || { echo "no last.pt to resume from: stopping"; break; }
    CK="--ckpt $LAST"
    echo "resuming from $LAST"
  fi
  CONDA_OVERRIDE_CUDA=12.8 CK="$CK" ROOT=$ROOT pixi run --frozen -e cu128 bash -c '
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 MW_COMPILE_TIMEOUT=1
    export PYTHONFAULTHANDLER=1
    export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld WANDB__SERVICE_WAIT=300
    export MW_SAMPLE_TIMING_DIR=/home/psk6950/MiniWorld-scratch/phase2a-v240/sample_timing_${SLURM_JOB_ID}; mkdir -p "$MW_SAMPLE_TIMING_DIR"
    export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_v240"; mkdir -p "$TRITON_CACHE_DIR"
    export PYTHONPATH=/home/psk6950/MiniWorld/scripts:${PYTHONPATH:-}
    torchrun --standalone --nproc_per_node=4 scripts/run_miniworld_diffusion_train.py train \
      --config configs/miniworld/phase2a_diffusion_medium_v240.yaml \
      --job-name v240-phase2a-medium-L384-local $CK --no-ckpt-strict \
      train.run_dir="$ROOT" train.use_wandb=true train.num_workers=32 train.prefetch_factor=16
  '
  rc=$?
  echo "==== attempt $attempt EXIT $rc ===="
  [ $rc -eq 0 ] && break
  now=$(stat -c %Y "$(ls -t "$ROOT"/*/*/checkpoints/last.pt 2>/dev/null | head -1)" 2>/dev/null || echo 0)
  [ "$now" = "$before" ] && { echo "no new checkpoint since the last attempt: stopping"; break; }
  before=$now
done
