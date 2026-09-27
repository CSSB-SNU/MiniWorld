#!/bin/bash
#SBATCH -J mw-ta-bn
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step17-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
python3 $R/install_variant.py pb-base2
# variant -> "BN MAXNREG"
declare -A VAR=( [bn128]="128 0" [bn64]="64 0" [bn32]="32 0" [nr128]="128 128" [bn64nr128]="64 128" [bn64nr96]="64 96" )
for V in bn128 bn64 bn32 nr128 bn64nr128 bn64nr96; do
  set -- ${VAR[$V]}
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_BN=$1 FPF_TRIATTPRO_MAXNREG=$2; \
    python -u -B $R/probe_regs.py --length 768" > $R/logs/regs-$V.log 2>&1
  echo "== regs $V"; grep -E "regs |ADDR=" $R/logs/regs-$V.log | head -3
done
for RD in r1 r2 r3; do
for V in bn128 bn64 bn32 nr128 bn64nr128 bn64nr96; do
  set -- ${VAR[$V]}
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_BN=$1 FPF_TRIATTPRO_MAXNREG=$2; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x7-$V-L$L-$RD.json" > $R/logs/x7-$V-L$L-$RD.log 2>&1
    grep -q "refused by kernels.triattn" $R/logs/x7-$V-L$L-$RD.log && echo "!! $V L$L $RD FELL BACK"
  done
done
done
python3 $R/summarise_ab.py x7 bn128 bn64 bn32 nr128 bn64nr128 bn64nr96 || true
echo DONE
