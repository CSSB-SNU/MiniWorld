#!/bin/bash
#SBATCH -J mw-ta-addr
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step16-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
python3 $R/install_variant.py pb-base2
for A in chain split; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_ADDR=$A; \
    python -u -B $R/probe_regs.py --length 768" > $R/logs/regs-$A.log 2>&1
  echo "== regs $A"; tail -3 $R/logs/regs-$A.log
done
for RD in r1 r2 r3; do
for A in chain split; do
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_ADDR=$A; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x6-$A-L$L-$RD.json" > $R/logs/x6-$A-L$L-$RD.log 2>&1
    if grep -q "refused by kernels.triattn" $R/logs/x6-$A-L$L-$RD.log; then
      echo "!! $A L$L $RD FELL BACK"
    fi
  done
done
done
python3 $R/summarise_ab.py x6 chain split || true
echo DONE
