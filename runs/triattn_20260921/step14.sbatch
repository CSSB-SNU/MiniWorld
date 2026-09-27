#!/bin/bash
#SBATCH -J mw-ta-ex2d
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step14-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for RD in r1 r2 r3; do
for T in base2 f16x2b; do
  python3 $R/install_variant.py pb-$T
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x4-$T-L$L-$RD.json" > $R/logs/x4-$T-L$L-$RD.log 2>&1
    # a run that fell back to flash is not a measurement of this kernel: say so loudly
    if grep -q "refused by kernels.triattn" $R/logs/x4-$T-L$L-$RD.log; then
      echo "!! $T L$L $RD FELL BACK: $(grep -oE 'refused by kernels.triattn \([^)]*\)' $R/logs/x4-$T-L$L-$RD.log | head -1 | cut -c1-120)"
    fi
  done
done
done
for T in base2 f16x2b; do for L in 384 768; do echo "== $T L$L"; grep -hE "^RESULT" $R/logs/x4-$T-L$L-r1.log | cut -c1-190 | head -1; done; done
echo DONE
