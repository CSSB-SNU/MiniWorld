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
# GPU-only step time of the phase-1a nr_nt setups, on the REAL train path (CUDA-graph trainer, Adam, EMA, grad accum, the real launcher's
# loader flags). +train.synth_data=true: the trainer reads a small pool of REAL batches once (real shapes / padding / MSA rows) and replays it,
# so no step waits for data. 1 GPU, GA micro-steps per optimizer step; the last lines print ms/sample = step_time / GA (median of the steady
# steps) and the mean data_wait (must be ~0, else the run is not GPU-only).
# env: PF (8|16), GA (default 16), EPOCHS (default 4), STEPS (optimizer steps per epoch, default 12), CFG (default the chainfix config)
set -uo pipefail
cd /home/psk6950/MiniWorld   # pixi env of the main checkout
REPO=${REPO:-/home/psk6950/MiniWorld}   # code + configs (a worktree for a patched trainer)
export REPO
: "${PF:?set PF=8 or 16}"
GA=${GA:-16}; EPOCHS=${EPOCHS:-4}; STEPS=${STEPS:-12}
ITEMS=$((GA * STEPS))
CFG=${CFG:-$REPO/configs/miniworld/phase1a_distogram_medium_v240_nr_nt_pf${PF}_chainfix.yaml}
RUN=/home/psk6950/MiniWorld-scratch/synth-speed/run_pf${PF}_${SLURM_JOB_ID}
mkdir -p "$RUN"
echo "=== synth speed PF=$PF GA=$GA epochs=$EPOCHS steps/epoch=$STEPS node $SLURM_JOB_NODELIST commit $(git -C $REPO rev-parse --short HEAD) cfg $CFG"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_v240"; mkdir -p "$TRITON_CACHE_DIR"
CONDA_OVERRIDE_CUDA=12.8 RUN="$RUN" PF=$PF GA=$GA EPOCHS=$EPOCHS ITEMS=$ITEMS CFG=$CFG pixi run --frozen -e cu128 bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_DEBUG=WARN MW_COMPILE_TIMEOUT=1 PYTHONFAULTHANDLER=1 PYTHONPATH=$REPO/src:$REPO/scripts:${PYTHONPATH:-}
  export MW_PACKED_LOADER=1 MW_COMPACT_MSA=1 MW_SKIP_UNREAD_STRUCTURE=1 MW_SKIP_SOURCES=
  torchrun --standalone --nproc_per_node=1 $REPO/scripts/run_miniworld_distogram_train.py train \
    --config $CFG --job-name synth-speed-pf${PF} \
    train.run_dir="$RUN" train.use_wandb=false +train.synth_data=true train.num_workers=8 train.prefetch_factor=4 \
    train.num_epoch=$EPOCHS train.train_item=$ITEMS train.grad_accum_steps=$GA train.save_freq=1000'
rc=$?
echo "==== EXIT $rc ===="
LOG=$(ls -t "$RUN"/*/*/train.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  grep "Step" "$LOG" | sed 's/.*step_time_ms=\([0-9.]*\) data_wait_ms=\([0-9.]*\).*/\1 \2/' | awk -v ga=$GA -v pf=$PF 'NR>3{a[++n]=$1; w+=$2} END{if(n){asort(a); printf "RESULT PF=%s GA=%s: %d steady steps, median step %.0f ms -> %.1f ms/sample (GPU-only), mean data_wait %.0f ms\n", pf, ga, n, a[int((n+1)/2)], a[int((n+1)/2)]/ga, w/n}}'
fi
