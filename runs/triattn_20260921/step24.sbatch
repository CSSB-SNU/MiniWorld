#!/bin/bash
#SBATCH -J mw-ta-bw
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step24-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 768 384; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0; python -u -B $R/bench_bandwidth.py --length $L --output $R/logs/bw-L$L.json" \
    > $R/logs/bw-L$L.log 2>&1
  echo "===== L$L"; grep -E "TB/s" $R/logs/bw-L$L.log || tail -15 $R/logs/bw-L$L.log
done
echo DONE
