#!/bin/bash
#SBATCH -J mw-bwd-core-followup
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/core90/build-pilot-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
if [[ "$KIND" == bias ]]; then
    FUSION_BUILD_ROOT="$ARTIFACT" FUSION_MODULE_NAME="triattn_bias_${ARTIFACT}" \
        bash "$ENV" python -u -B "$R/bias_fusion/native.py" 2>&1 | tee "$R/bias_fusion/$ARTIFACT/build.log"
elif [[ "$KIND" == dq ]]; then
    DQ_BUILD_ROOT="$ARTIFACT" DQ_MODULE_NAME="triattn_dq_${ARTIFACT}" \
        bash "$ENV" python -u -B "$R/dq/native.py" 2>&1 | tee "$R/dq/$ARTIFACT/build.log"
else
    exit 2
fi
for L in 64 384 768 1024; do
    bash "$ENV" python -u -B "$R/below80/check.py" --kind "$KIND" --artifact "$ARTIFACT" \
        --length "$L" --output "$R/core90/pilot-${SLURM_JOB_ID}-L${L}.json"
done
