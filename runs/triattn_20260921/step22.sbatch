#!/bin/bash
#SBATCH -J mw-ta-epi
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step22-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
python3 $R/install_variant.py pb-base2
# The block reads the CELL row (pair_fused_cells.json r16), not epilogue.py's PINNED_CONFIG, so the tile and the register
# cap are both swept here through the two env overrides that DO reach it.  Prologue held at its measured best throughout.
PRO="FPF_TRIATTPRO_BN=64 FPF_TRIATTPRO_MAXNREG=128"
declare -A VAR=(
  [ref]="- - -"
  [t168n0]="16,8,4,1 0 $PRO"    [t168n200]="16,8,4,1 200 $PRO"  [t168n160]="16,8,4,1 160 $PRO"
  [t88n0]="8,8,4,1 0 $PRO"      [t88n200]="8,8,4,1 200 $PRO"    [t88n160]="8,8,4,1 160 $PRO"   [t88n224]="8,8,4,1 224 $PRO"
  [t164n0]="16,4,4,1 0 $PRO"    [t164n200]="16,4,4,1 200 $PRO"
  [t816n0]="8,16,4,1 0 $PRO"    [t88w8n0]="8,8,8,1 0 $PRO"
)
ORDER="ref t168n0 t168n200 t168n160 t88n0 t88n200 t88n160 t88n224 t164n0 t164n200 t816n0 t88w8n0"
for RD in r1 r2 r3; do
for V in $ORDER; do
  A=(${VAR[$V]}); EXTRA=""
  [ "${A[0]}" != "-" ] && EXTRA="FPF_TRIATT_EPI_CFG=${A[0]} FPF_TRIATT_EPI_MAXNREG=${A[1]} ${A[@]:2}"
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH $EXTRA; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/xa-$V-L$L-$RD.json" > $R/logs/xa-$V-L$L-$RD.log 2>&1
    grep -q "refused by kernels.triattn" $R/logs/xa-$V-L$L-$RD.log && echo "!! $V L$L $RD FELL BACK"
  done
done
done
python3 $R/summarise_ab.py xa $ORDER || true
echo DONE
