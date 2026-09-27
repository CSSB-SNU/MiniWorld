#!/bin/bash
#SBATCH -J mw-qkv-sanitize
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/qkv-sanitize-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for L in ${LENGTHS:-128 768}; do
    for TOOL in memcheck racecheck synccheck; do
        timeout 480s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=qkv_attention_resident' --error-exitcode 86 python -u -B "$R/qkv_check.py" --artifact "$ARTIFACT" --length "$L" --native-only --cases mixed all_masked --output "$R/qkv-${TOOL}-${SLURM_JOB_ID}-L${L}.json"
    done
done
