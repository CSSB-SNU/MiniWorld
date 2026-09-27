#!/bin/bash
#SBATCH -J mw-ta-epi
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step7-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --ln stock --output $R/logs/ln2-stock-L$L.json" > $R/logs/ln2-stock-L$L.log 2>&1
    echo "-- ln=stock L$L"; grep -hE "^RESULT|Unsupported|Error" $R/logs/ln2-stock-L$L.log | cut -c1-190
    for C in cell 16,8,4,1 8,16,4,1 16,16,4,1 8,8,4,1 16,8,8,1 16,8,4,2 32,8,4,1 16,4,4,1 16,8,4,1,3 8,16,8,1; do
      TAG=$(echo $C | tr ',' '_')
      if [ "$C" = "cell" ]; then E=""; else E="export FPF_TRIATT_EPI_CFG=$C;"; fi
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; $E \
        python -u -B $R/sweep_pro.py --length $L --output $R/logs/epi2-$TAG-L$L.json" > $R/logs/epi2-$TAG-L$L.log 2>&1
      grep -hE "^RESULT" $R/logs/epi2-$TAG-L$L.log | cut -c1-200 || echo "  epi=$C L$L failed: $(grep -oE '[A-Za-z]*Error.*' $R/logs/epi2-$TAG-L$L.log | head -1 | cut -c1-80)"
    done
  ) &
done
wait
echo DONE
