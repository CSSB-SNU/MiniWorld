#!/bin/bash
#SBATCH -J mw-ta-resid
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step2-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
# env.sh puts the pristine opt_core on PYTHONPATH; put the patched copy FIRST so `opt_core` resolves to it.
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  ( bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B -c \"import opt_core,sys;print('opt_core from',opt_core.__file__)\" && \
      python -u -B $R/bench_residual.py --length $L --output $R/logs/resid-L$L.json" > $R/logs/resid-L$L.log 2>&1
    echo "== L$L"; grep -hE "^RESULT|opt_core from|Traceback|Error" $R/logs/resid-L$L.log | cut -c1-300 ) &
done
wait
echo DONE
