#!/bin/bash
#SBATCH --job-name=mw_v240_p1a_med_scratch
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
#SBATCH --output=/home/psk6950/MiniWorld-scratch/phase1a-v240/train_%j.out
#SBATCH --error=/home/psk6950/MiniWorld-scratch/phase1a-v240/train_%j.out
# MiniWorld v2.4.0 phase 1a, medium, L384, FROM SCRATCH, data on node02's local NVMe (/local_data1), epochs chained (no loader reset at the
# epoch boundary), W&B on forge team_gm/MiniWorld as a new run. A crash restarts from this run's own last.pt (up to 5 times).
set -uo pipefail
cd /home/psk6950/MiniWorld
ROOT=/home/psk6950/MiniWorld/runs/v2.4.0/phase1a/medium_distogram_L384_scratch
echo "=== job $SLURM_JOB_ID node $SLURM_JOB_NODELIST  commit $(git rev-parse --short HEAD)  root $ROOT"
before=""
for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt  $(date)"
  if [ "$attempt" = 1 ]; then
    RESUME=""
  else
    RUN=$(sed -n 1p "$ROOT/latest_run.txt")
    RESUME="--ckpt $RUN/checkpoints/last.pt +train.resume_run_subdir=$RUN"
    echo "resuming $RUN"
  fi
  CONDA_OVERRIDE_CUDA=12.8 RESUME="$RESUME" ROOT=$ROOT pixi run --frozen -e cu128 bash -c '
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 MW_COMPILE_TIMEOUT=1
    export PYTHONFAULTHANDLER=1
    export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld WANDB__SERVICE_WAIT=300
    export MW_SAMPLE_TIMING_DIR=/home/psk6950/MiniWorld-scratch/phase1a-v240/sample_timing_${SLURM_JOB_ID}; mkdir -p "$MW_SAMPLE_TIMING_DIR"
    export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_v240"; mkdir -p "$TRITON_CACHE_DIR"
    export PYTHONPATH=/home/psk6950/MiniWorld/scripts:${PYTHONPATH:-}
    if [ -n "$RESUME" ]; then export MW_WANDB_NEW_RUN=1; fi
    torchrun --standalone --nproc_per_node=4 scripts/run_miniworld_distogram_train.py train \
      --config configs/miniworld/phase1a_distogram_medium_v240_local.yaml \
      --job-name v240-phase1a-medium-L384-local-scratch $RESUME \
      train.run_dir="$ROOT" train.use_wandb=true train.num_workers=32 train.prefetch_factor=16
  '
  rc=$?
  echo "==== attempt $attempt EXIT $rc ===="
  [ $rc -eq 0 ] && break
  now=$(stat -c %Y "$(sed -n 1p "$ROOT/latest_run.txt")/checkpoints/last.pt" 2>/dev/null || echo 0)
  [ "$now" = "$before" ] && { echo "no new checkpoint since the last attempt: stopping"; break; }
  before=$now
done
