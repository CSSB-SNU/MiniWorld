#!/bin/bash
#SBATCH -J mw-ctaheads-audit
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference/cta-heads-audit-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_inference
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$R/cta_heads_front_pair.py" --output "$R/cta-heads-front-pair-${SLURM_JOB_ID}.json"
for A in cta_heads4_s2 cta_heads4_repack; do
  for L in 64 128 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/hot_audit.py" --artifact "$A" --length "$L" --output "$R/audit-${A}-${L}-${SLURM_JOB_ID}.json"
  done
  for TOOL in memcheck racecheck synccheck; do
    for CASE in mixed all_masked large_logits; do
      timeout 300s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=qg_attention_cta_heads4|head4_kv_projection' --error-exitcode 86 python -u -B "$R/check.py" --artifact "$A" --length 128 --native-only --cases "$CASE" --output "$R/${TOOL}-${A}-128-${CASE}-${SLURM_JOB_ID}.json"
    done
  done
  timeout 360s bash "$ENV" python -u -B "$R/module_check.py" --front --front-artifact front8 --qualify --artifact "$A" --length 384 --rounds 16 --replays 20 --output "$R/qualified-${A}-384-${SLURM_JOB_ID}.json"
done
for TOOL in memcheck racecheck synccheck; do
  timeout 300s bash "$ENV" compute-sanitizer --tool "$TOOL" --error-exitcode 86 python -u -B "$R/front_check.py" --artifact front8_hlast --length 128 --native-only --output "$R/${TOOL}-front8_hlast-128-${SLURM_JOB_ID}.json"
done
