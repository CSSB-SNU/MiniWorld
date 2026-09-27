#!/bin/bash
#SBATCH -J mw-ta-ex2b
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step11-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
PB=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt
for RD in r1 r2; do
for T in base f16x2; do
  rm -rf $PB && cp -r $R/prebuilt-$T $PB
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x2-$T-L$L-$RD.json" > $R/logs/x2-$T-L$L-$RD.log 2>&1
    echo "== $T L$L $RD"; grep -hE "^RESULT|refused|Error" $R/logs/x2-$T-L$L-$RD.log | cut -c1-230 | head -2
  done
done
done
echo DONE
