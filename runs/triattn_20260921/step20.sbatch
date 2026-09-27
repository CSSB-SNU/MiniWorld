#!/bin/bash
#SBATCH -J mw-ta-occ2
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step20-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
python3 $R/install_variant.py pb-base2
# every run keeps the prologue at its new best (BN 64 + maxnreg 128) and moves only the epilogue's register cap,
# then the reverse: epilogue at its best and the prologue pushed past 2 CTA/SM with 4-warp CTAs.
# name -> "PRO_BN PRO_NREG PRO_CFG EPI_NREG"
declare -A VAR=(
  [ref]="128 0 cell 0"          [p0]="64 128 cell 0"
  [e255]="64 128 cell 200"      [e160]="64 128 cell 160"   [e128]="64 128 cell 128"
  [e96]="64 128 cell 96"        [e72]="64 128 cell 72"     [e64]="64 128 cell 64"
  [w4n96]="64 96 8,16,4,2 0"    [w4n80]="64 80 8,16,4,2 0" [w4n64]="64 64 8,16,4,2 0"
  [b32n80]="32 80 8,16,4,2 0"   [b32n64]="32 64 8,16,4,2 0"
)
ORDER="ref p0 e255 e160 e128 e96 e72 e64 w4n96 w4n80 w4n64 b32n80 b32n64"
for RD in r1 r2 r3; do
for V in $ORDER; do
  set -- ${VAR[$V]}
  CFG=""; [ "$3" != "cell" ] && CFG="FPF_TRIATTPRO_CFG=$3"
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH \
      FPF_TRIATTPRO_BN=$1 FPF_TRIATTPRO_MAXNREG=$2 $CFG FPF_TRIATT_EPI_MAXNREG=$4; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x9-$V-L$L-$RD.json" > $R/logs/x9-$V-L$L-$RD.log 2>&1
    grep -q "refused by kernels.triattn" $R/logs/x9-$V-L$L-$RD.log && echo "!! $V L$L $RD FELL BACK"
  done
done
done
python3 $R/summarise_ab.py x9 $ORDER || true
echo DONE
