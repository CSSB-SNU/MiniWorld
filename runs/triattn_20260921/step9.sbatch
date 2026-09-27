#!/bin/bash
#SBATCH -J mw-ta-ex2
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step9-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export TORCH_EXTENSIONS_DIR=$R/ext
mkdir -p $R/ext
for L in 768 384; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    for D in "" "TRIATTN_EX2_F16X2=1"; do
      TAG=$([ -z "$D" ] && echo base || echo f16x2)
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH TORCH_EXTENSIONS_DIR=$R/ext TRIATTN_M1_DEFINES='$D'; \
        python -u -B $R/sweep_pro.py --length $L --output $R/logs/ex2-$TAG-L$L.json" > $R/logs/ex2-$TAG-L$L.log 2>&1
      echo "== $TAG L$L"; grep -hE "^RESULT|error|Error" $R/logs/ex2-$TAG-L$L.log | cut -c1-230 | head -3
      grep -hE "registers|spill" $R/logs/ex2-$TAG-L$L.log | grep -i triattn_m1 | head -2
    done
  ) &
done
wait
echo DONE
