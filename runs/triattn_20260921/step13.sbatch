#!/bin/bash
#SBATCH -J mw-ta-pro-dev
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step13-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    for V in 0 1; do
      TAG=$([ $V = 0 ] && echo asm || echo noasm)
      for RD in r1 r2; do
        bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH FPF_TRIATTPRO_NOASM=$V TRITON_CACHE_DIR=$R/tc-$TAG; \
          python -u -B $R/sweep_pro.py --length $L --output $R/logs/pro-$TAG-L$L-$RD.json" > $R/logs/pro-$TAG-L$L-$RD.log 2>&1
      done
      echo "== $TAG L$L"; grep -hE "^RESULT|Error" $R/logs/pro-$TAG-L$L-r1.log | cut -c1-210 | head -2
    done
  ) &
done
wait
echo DONE
