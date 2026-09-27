#!/bin/bash
#SBATCH -J mw-ta-qkv
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step26-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
# q/k/v physical layout: "head" = [I,H,J,D] storage (64 B runs), "pair" = [I,J,H,D] storage handed as the [I,H,J,D] view
# (one H*D = 256 B run per (i,j)).  Measured separately: the core is within 0.3 % either way at L >= 768.
BEST="FPF_TRIATTPRO_BN=64 FPF_TRIATTPRO_MAXNREG=128 FPF_TRIATT_EPI_CFG=8,8,4,1"
for RD in r1 r2 r3; do
for QK in head pair; do
  for L in 384 768 1024; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH $BEST FPF_TRIATTPRO_QKV=$QK; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/xb-$QK-L$L-$RD.json" > $R/logs/xb-$QK-L$L-$RD.log 2>&1
    grep -q "refused by kernels.triattn" $R/logs/xb-$QK-L$L-$RD.log && echo "!! $QK L$L $RD FELL BACK"
  done
done
done
python3 $R/summarise_ab.py xb head pair || true
echo DONE
