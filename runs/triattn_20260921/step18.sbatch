#!/bin/bash
#SBATCH -J mw-ta-flags
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step18-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
# 8 = PRMT pack (P -> bf16 by truncation on the ALU pipe instead of F2FP on the XU pipe, which MUFU also uses)
# 32 = E-phase token ring   1048576 = Q in registers   4194304 = kW   8388608 / 16777216 = one-time 200 / 400 cycle de-phasing of the 3 consumer warpgroups
FL="0;8;32;1048576;4194304;8388608;16777216;8388616"
CFL="0,8,32,1048576,4194304,8388608,16777216,8388616"
for L in 768 384; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 TRIATTN_M1_FLAGS='$FL' \
    CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 \
    TORCH_EXTENSIONS_DIR=$R/ext-flags MAX_JOBS=16; \
    python -u -B $R/bench_core_flags.py --length $L --flags $CFL --output $R/logs/coreflags-L$L.json" \
    > $R/logs/coreflags-L$L.log 2>&1
  echo "===== L$L"; grep -E "^flags" $R/logs/coreflags-L$L.log || tail -25 $R/logs/coreflags-L$L.log
done
echo DONE
