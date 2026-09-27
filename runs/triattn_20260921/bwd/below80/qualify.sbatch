#!/bin/bash
#SBATCH -J mw-below80-qualified
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80/qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd/below80
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/check_module.py"
if [[ "$KIND" == bias ]]; then
  FUSION_ARTIFACT="$ARTIFACT" FUSION_ROWS=4 FUSION_ROW_OPTIONS=4 bash "$ENV" python -u -B "$R/../bias_fusion/check.py" --output "$R/fp64-${SLURM_JOB_ID}.json"
  FUSION_ARTIFACT="$ARTIFACT" FUSION_ROWS=4 bash "$ENV" python -u -B "$R/../bias_fusion/check_cancellation.py"
fi
for TOOL in memcheck racecheck synccheck; do
  bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=grouped_dkdv|reduce_bias|dq_tma|projection_ln_residual_tma|reduce_norm|finish_norm' --error-exitcode 86 python -u -B "$R/check.py" --kind "$KIND" --artifact "$ARTIFACT" --length 64 --native-only --output "$R/${TOOL}-${SLURM_JOB_ID}.json"
done
