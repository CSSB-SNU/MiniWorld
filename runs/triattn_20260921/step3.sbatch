#!/bin/bash
#SBATCH -J mw-ta-pro
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step3-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
CFGS="cell 8,16,8,2 8,16,4,2 8,16,8,3 8,16,4,3 4,16,4,2 4,32,4,2 16,16,8,2 8,32,8,2 8,32,4,2 16,32,8,2 4,8,4,2 8,8,4,2 16,8,8,2 32,16,8,2 8,64,8,2 4,16,4,3"
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    for C in $CFGS; do
      TAG=$(echo $C | tr ',' '_')
      if [ "$C" = "cell" ]; then E=""; else E="export FPF_TRIATTPRO_CFG=$C;"; fi
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; $E \
        python -u -B $R/sweep_pro.py --length $L --output $R/logs/pro-$TAG-L$L.json" > $R/logs/pro-$TAG-L$L.log 2>&1
      grep -hE "^RESULT" $R/logs/pro-$TAG-L$L.log | cut -c1-170 || echo "  $C L$L FAILED: $(grep -oE '[A-Za-z]*Error.*' $R/logs/pro-$TAG-L$L.log | head -1 | cut -c1-90)"
    done
  ) &
done
wait
echo DONE
