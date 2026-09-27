#!/bin/bash
#SBATCH -J mw-ta-core
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step6-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    # (4) the core is a parameter of the block: try every selector the block accepts
    for C in tier:triattn_native default flash_triattn k2b tier:k2b tier:flash tier:cuda_sm90a tier:fast; do
      TAG=$(echo $C | tr ':' '-')
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; \
        python -u -B $R/sweep_pro.py --length $L --core $C --output $R/logs/core-$TAG-L$L.json" > $R/logs/core-$TAG-L$L.log 2>&1
      grep -hE "^RESULT" $R/logs/core-$TAG-L$L.log | cut -c1-190 || echo "  core=$C L$L refused: $(grep -oE '(Unsupported|Refusal|[A-Za-z]*Error).*' $R/logs/core-$TAG-L$L.log | head -1 | cut -c1-90)"
    done
    # (2b) the prologue's other structural option: LN split out of the kernel
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --ln stock --output $R/logs/lnstock-L$L.json" > $R/logs/lnstock-L$L.log 2>&1
    echo "-- ln=stock L$L"; grep -hE "^RESULT|Unsupported|Error" $R/logs/lnstock-L$L.log | cut -c1-190
    # (3) epilogue config sweep
    for C in cell 2,16,8,4,1 2,8,8,4,1 2,16,16,4,1 2,32,8,4,1 2,16,8,8,1 2,16,8,4,2 2,8,16,4,1 3,16,8,4,1; do
      TAG=$(echo $C | tr ',' '_')
      if [ "$C" = "cell" ]; then E=""; else E="export FPF_TRIATT_EPI_CFG=$C;"; fi
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R/oc:\$PYTHONPATH; $E \
        python -u -B $R/sweep_pro.py --length $L --output $R/logs/epi-$TAG-L$L.json" > $R/logs/epi-$TAG-L$L.log 2>&1
      grep -hE "^RESULT" $R/logs/epi-$TAG-L$L.log | cut -c1-190 || echo "  epi=$C L$L failed"
    done
  ) &
done
wait
echo DONE
