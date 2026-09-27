#!/bin/bash
#SBATCH -J mw-qkv-attn-module
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/qkv-module-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
EXTRA=()
if [[ "${STAGE:-0}" == 1 ]]; then EXTRA+=(--stage); fi
if [[ "${INSTALLED:-0}" == 1 ]]; then EXTRA+=(--installed); fi
for L in ${LENGTHS:-384 768 1024}; do
    timeout 420s bash "$ENV" python -u -B "$R/qkv_module_check.py" --artifact "$ARTIFACT" --length "$L" --mode "${MODE:-bench}" --rounds "${ROUNDS:-16}" --replays "${REPLAYS:-20}" "${EXTRA[@]}" --output "$R/qkv-module-${SLURM_JOB_ID}-L${L}.json"
done
