#!/bin/bash
#SBATCH -J mw-ta-ex2c
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step12-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
TOOLS=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/tools
PB=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt
CUT=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2
for D in "" "TRIATTN_EX2_F16X2=1"; do
  TAG=$([ -z "$D" ] && echo base2 || echo f16x2b)
  rm -rf $R/ext-$TAG $R/pb-$TAG; mkdir -p $R/ext-$TAG
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH TORCH_EXTENSIONS_DIR=$R/ext-$TAG \
    CUTLASS_PATH=$CUT TRIATTN_M1_DEFINES='$D'; cd $TOOLS && python -u build_prebuilt.py --exts triattn_m1_ext" \
    > $R/logs/b2-$TAG.log 2>&1
  echo "== build $TAG"; grep -E "Used .* registers|BUILD_RESULTS" $R/logs/b2-$TAG.log | tail -2 | cut -c1-150
  cp -r $PB $R/pb-$TAG
  python3 -c "
import json;d=json.load(open('$R/pb-$TAG/torch2.10.0+cu128-cpython-310-x86_64-linux-gnu/triattn_m1_ext.json'));print('   so',d['so_sha256'][:16],d['built_utc'])"
done
for RD in r1 r2 r3; do
for T in base2 f16x2b; do
  rm -rf $PB && cp -r $R/pb-$T $PB
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x3-$T-L$L-$RD.json" > $R/logs/x3-$T-L$L-$RD.log 2>&1
  done
done
done
for T in base2 f16x2b; do for L in 384 768; do echo "== $T L$L"; grep -hE "^RESULT|refused" $R/logs/x3-$T-L$L-r1.log | cut -c1-200 | head -1; done; done
echo DONE
