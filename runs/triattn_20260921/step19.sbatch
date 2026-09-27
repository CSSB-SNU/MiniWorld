#!/bin/bash
#SBATCH -J mw-ta-occ
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step19-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
python3 $R/install_variant.py pb-base2
# maxnreg at 256 threads: <=128 -> 2 CTA/SM, <=85 -> 3 CTA/SM.  The tile config was pinned at 1 CTA/SM, so it is re-opened here.
# name -> "BN MAXNREG CFG(BI,BJ,W,S)"
declare -A VAR=(
  [base]="128 0 cell"      [a]="64 128 cell"        [b]="128 128 cell"
  [c]="64 120 cell"        [d]="64 112 cell"        [e]="64 85 cell"
  [f]="64 128 8,16,8,3"    [g]="64 128 8,16,8,4"    [h]="64 128 8,16,4,2"
  [i]="64 128 16,16,8,2"   [j]="64 128 8,32,8,2"    [k]="64 128 8,8,8,2"
)
ORDER="base a b c d e f g h i j k"
for RD in r1 r2 r3; do
for V in $ORDER; do
  set -- ${VAR[$V]}
  CFG=""; [ "$3" != "cell" ] && CFG="FPF_TRIATTPRO_CFG=$3"
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_BN=$1 FPF_TRIATTPRO_MAXNREG=$2 $CFG; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x8-$V-L$L-$RD.json" > $R/logs/x8-$V-L$L-$RD.log 2>&1
    grep -q "refused by kernels.triattn" $R/logs/x8-$V-L$L-$RD.log && echo "!! $V L$L $RD FELL BACK"
  done
done
done
python3 $R/summarise_ab.py x8 $ORDER || true
echo DONE
