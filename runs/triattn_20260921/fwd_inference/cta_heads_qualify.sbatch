#!/bin/bash
#SBATCH -J mw-ctaheads-qualify
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:50:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/cta-heads-qualify-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${ARTIFACT:?artifact required}"
for L in ${AUDIT_LENGTHS:-64 128 384 768 1024}; do
  timeout 180s bash "$ENV" python -u -B "$R/hot_audit.py" --artifact "$ARTIFACT" --length "$L" --output "$R/audit-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
for L in ${SAN_LENGTHS:-128 384 768 1024}; do
  for TOOL in memcheck racecheck synccheck; do
    for CASE in mixed all_masked large_logits; do
      timeout 600s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=qg_attention_cta_heads4|head4_kv_projection' --error-exitcode 86 python -u -B "$R/check.py" --artifact "$ARTIFACT" --length "$L" --native-only --cases "$CASE" --output "$R/${TOOL}-${ARTIFACT}-${L}-${CASE}-${SLURM_JOB_ID}.json"
    done
    timeout 300s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 86 python -u -B "$R/front_check.py" --artifact front8_hlast --length "$L" --native-only --output "$R/${TOOL}-front8_hlast-${L}-${SLURM_JOB_ID}.json"
  done
done
for L in ${LENGTHS:-384 768 1024}; do
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact front8 --qualify --artifact "$ARTIFACT" --length "$L" --rounds 24 --replays 30 --output "$R/qualified-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
  timeout 360s bash "$ENV" python -u -B "$R/anthropic_compare.py" --artifact "$ARTIFACT" --previous --previous-artifact selected --length "$L" --rounds 64 --replays 40 --output "$R/anthropic-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
