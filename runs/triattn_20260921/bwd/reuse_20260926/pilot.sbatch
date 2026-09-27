#!/bin/bash
#SBATCH -J mw-bwd-reuse
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926/pilot-%j.log
set -euo pipefail
B=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
R=$B/reuse_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${KIND:?}" "${ARTIFACTS:?}"
for A in $ARTIFACTS; do
  if [[ $KIND == dq ]]; then
    DQ_BUILD_ROOT="$A" DQ_MODULE_NAME="triattn_dq_$A" bash "$ENV" python -u -B "$B/dq/native.py" 2>&1 | tee "$B/dq/$A/build.log"
    DIR=$B/dq/$A; LIB=triattn_dq
  else
    FUSION_BUILD_ROOT="$A" FUSION_MODULE_NAME="triattn_bias_$A" bash "$ENV" python -u -B "$B/bias_fusion/native.py" 2>&1 | tee "$B/bias_fusion/$A/build.log"
    DIR=$B/bias_fusion/$A; LIB=triattn_bias_fusion
  fi
  bash "$ENV" cuobjdump --dump-resource-usage "$DIR/build/$LIB.so" > "$DIR/resources.txt"
  bash "$ENV" cuobjdump --dump-sass "$DIR/build/$LIB.so" > "$DIR/sass.txt"
  for L in 64 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/check_native.py" --kind "$KIND" --artifact "$A" --length "$L" --output "$R/native-${A}-${L}-${SLURM_JOB_ID}.json"
  done
done
