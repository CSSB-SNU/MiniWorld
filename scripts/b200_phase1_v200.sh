#!/usr/bin/env bash
# MiniWorld v2.0.0 phase 1 (medium trunk, distogram CE) on B200 GPUs.
#
#   scripts/b200_phase1_v200.sh 1a                       # L384, fresh run
#   scripts/b200_phase1_v200.sh 1b                       # L768, continues the 1a run from its last.pt
#   RESUME=<run subdir> scripts/b200_phase1_v200.sh 1a   # resume an interrupted run from <subdir>/checkpoints/last.pt
#   SMOKE=1 scripts/b200_phase1_v200.sh 1a               # 2 short epochs, no W&B, scratch run_dir: end-to-end check + step time
#   DRYRUN=1 scripts/b200_phase1_v200.sh 1a              # print the training command and stop
#
# Node 1 (default): GPUS (default 4,5,6,7) are locked one gpuq job each: the first GPU runs the training, the others hold their
# lock until it ends, so the gpuq fillers (dummy matmuls) cannot touch them.
# Node 2 (NODE=2, the 8-GPU box, same Lustre and the same $W): no gpuq (its state folder is shared with node 1, locks would mix),
# GPUS defaults to 0,1,2,3,4,5,6,7 and torchrun runs on them directly. Run it detached there, e.g.
#   NODE=2 setsid nohup scripts/b200_phase1_v200.sh 1a > $W/logs/phase1a_node2.log 2>&1 &
# The effective batch stays 256: train.grad_accum_steps = 256 / (number of GPUs) (64 on 4 GPUs, 32 on 8) unless you pass your own.
# Caches (Triton, inductor, the engine's CUDA extensions, TMPDIR: /tmp is noexec on both nodes) come from $W/pixi_env.sh.
# W&B goes to forge.coreweave.com, team_gm / MiniWorld, with the key from ~/.netrc (WANDB_BASE_URL selects the right entry).
set -euo pipefail

PHASE=${1:?usage: b200_phase1_v200.sh 1a|1b}
shift || true
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
NODE=${NODE:-1}
case $NODE in
  1) GPUS=${GPUS:-4,5,6,7}; USE_GPUQ=1 ;;
  2) GPUS=${GPUS:-0,1,2,3,4,5,6,7}; USE_GPUQ=0 ;;
  *) echo "NODE must be 1 or 2" >&2; exit 2 ;;
esac
IFS=, read -ra G <<<"$GPUS"
NGPU=${#G[@]}
if (( 256 % NGPU != 0 )); then echo "the effective batch 256 must be divisible by the number of GPUs ($NGPU)" >&2; exit 2; fi
REPO=${REPO:-$W/MiniWorld}
GPUQ=$W/gpuq/gpuq
case $PHASE in
  1a) CONFIG=phase1a_distogram_medium_v200_b200 ;;
  1b) CONFIG=phase1b_distogram_medium_v200_b200 ;;
  *) echo "phase must be 1a or 1b" >&2; exit 2 ;;
esac
RUN_DIR=$W/runs/b200/phase1_v200/medium_distogram
NAME=v2.0.0-phase${PHASE}-medium-b200
EXTRA=("$@")
case " ${EXTRA[*]-} " in
  *" train.grad_accum_steps="*) ;;
  *) EXTRA=("train.grad_accum_steps=$((256 / NGPU))" ${EXTRA[@]+"${EXTRA[@]}"}) ;;
esac
if [[ -n ${SMOKE:-} ]]; then
  RUN_DIR=$W/scratch/p1_smoke_$PHASE
  NAME=smoke-$NAME
  EXTRA+=(train.train_item=$((NGPU * 256)) train.valid_item=$((NGPU * 8)) train.num_epoch=2 train.use_wandb=false)
  export WANDB_MODE=disabled
fi

# resume source: RESUME=<run subdir>; phase 1b always continues the run of phase 1a (the newest subdir of RUN_DIR)
CKPT_ARGS=()
if [[ -n ${RESUME:-} ]]; then
  export MW_RESUME_RUN_SUBDIR=$RESUME
  CKPT_ARGS=(--ckpt "$RESUME/checkpoints/last.pt")
elif [[ $PHASE == 1b ]]; then
  SUB=$(ls -dt "$RUN_DIR"/*/*/ 2>/dev/null | head -1 || true)
  [[ -n $SUB && -f ${SUB%/}/checkpoints/last.pt ]] || { echo "no phase 1a run with checkpoints/last.pt under $RUN_DIR" >&2; exit 3 ;}
  export MW_RESUME_RUN_SUBDIR=${SUB%/}
  CKPT_ARGS=(--ckpt "${SUB%/}/checkpoints/last.pt")
fi

TRAIN=$(cat <<EOS
set -euo pipefail
source $W/pixi_env.sh
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$REPO/src:$REPO/scripts PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld WANDB_DIR=$W/scratch
cd $REPO
exec pixi run --manifest-path $W/mwenv/pixi.toml python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPU scripts/run_miniworld_distogram_train.py train --config configs/miniworld/$CONFIG.yaml ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} --job-name $NAME train.engine_backend=auto "train.run_dir=$RUN_DIR" ${EXTRA[@]+"${EXTRA[@]}"}
EOS
)
if [[ -n ${DRYRUN:-} ]]; then
  echo "NODE=$NODE USE_GPUQ=$USE_GPUQ GPUS=$GPUS"
  echo "$TRAIN"
  exit 0
fi

if [[ $USE_GPUQ == 0 ]]; then
  exec bash -c "$TRAIN"
fi

STATE=$W/scratch/p1_launch_$$; mkdir -p "$STATE"
cleanup() { touch "$STATE/done"; wait 2>/dev/null || true; }
trap cleanup EXIT

# hold the other GPUs
for g in "${G[@]:1}"; do
  "$GPUQ" run -g "$g" -n "$NAME-hold$g" -- bash -c "touch $STATE/held$g; until [ -e $STATE/done ]; do sleep 20; done" &
done
for g in "${G[@]:1}"; do
  until [ -e "$STATE/held$g" ]; do sleep 2; done
done
echo "GPUs held: ${GPUS}" >&2
"$GPUQ" run -g "${G[0]}" -n "$NAME" -- bash -c "$TRAIN"
