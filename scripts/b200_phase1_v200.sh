#!/usr/bin/env bash
# MiniWorld v2.0.0 phase 1 (medium trunk, distogram CE) on 4 B200 GPUs, every GPU held through gpuq.
#
#   scripts/b200_phase1_v200.sh 1a                       # L384, fresh run
#   scripts/b200_phase1_v200.sh 1b                       # L768, continues the 1a run from its last.pt
#   RESUME=<run subdir> scripts/b200_phase1_v200.sh 1a   # resume an interrupted run from <subdir>/checkpoints/last.pt
#   SMOKE=1 scripts/b200_phase1_v200.sh 1a               # 2 short epochs, no W&B, scratch run_dir: end-to-end check + step time
#
# GPUS (default 4,5,6,7) are locked one gpuq job each: the first GPU runs the training, the others hold their lock until it
# ends, so the gpuq fillers (dummy matmuls) cannot touch them. W&B goes to forge.coreweave.com, team_gm / MiniWorld, with the
# key from ~/.netrc (WANDB_BASE_URL is what selects the right entry; never WANDB_MODE=offline here).
set -euo pipefail

PHASE=${1:?usage: b200_phase1_v200.sh 1a|1b}
shift || true
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
GPUS=${GPUS:-4,5,6,7}
IFS=, read -ra G <<<"$GPUS"
NGPU=${#G[@]}
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

TRAIN=$(cat <<EOS
set -euo pipefail
source $W/pixi_env.sh
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$REPO/src:$REPO/scripts PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
# the B200 /tmp is mounted noexec: Triton cannot load its compiled helper modules from there
export TRITON_CACHE_DIR=$W/scratch/cache/triton_p1 TORCHINDUCTOR_CACHE_DIR=$W/scratch/cache/inductor_p1
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=team_gm WANDB_PROJECT=MiniWorld WANDB_DIR=$W/scratch
cd $REPO
exec pixi run --manifest-path $W/mwenv/pixi.toml python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPU \\
  scripts/run_miniworld_distogram_train.py train \\
  --config configs/miniworld/$CONFIG.yaml ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} --job-name $NAME \\
  train.engine_backend=auto +train.force_trainer=fabric train.compile=true "train.run_dir=$RUN_DIR" ${EXTRA[@]+"${EXTRA[@]}"}
EOS
)
"$GPUQ" run -g "${G[0]}" -n "$NAME" -- bash -c "$TRAIN"
