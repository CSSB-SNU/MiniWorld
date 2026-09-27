#!/bin/bash
#SBATCH -J mw-ta-rebuild
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step10-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
TOOLS=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/tools
export TORCH_EXTENSIONS_DIR=$R/ext
mkdir -p $R/ext
# The face verifies the prebuilt .so against the sources; a modified source needs a new record from the
# producer's own tool.  Build the patched tree twice: switch off (must reproduce the stock numbers) and on.
for D in "" "TRIATTN_EX2_F16X2=1"; do
  TAG=$([ -z "$D" ] && echo base || echo f16x2)
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH TORCH_EXTENSIONS_DIR=$R/ext \
    CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 TRIATTN_M1_DEFINES='$D'; cd $TOOLS && \
    python -u build_prebuilt.py --exts triattn_m1_ext" > $R/logs/build-$TAG.log 2>&1
  echo "== build $TAG rc=$?"; tail -6 $R/logs/build-$TAG.log | cut -c1-200
  cp -r $R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt $R/prebuilt-$TAG 2>/dev/null
done
echo DONE
