#!/bin/bash
#SBATCH -J mw-bwd-reuse-qual
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/bwd/reuse_20260926/qualify-%j.log
set -euo pipefail
B=/home/psk6950/MiniWorld/runs/triattn_20260921/bwd
R=$B/reuse_20260926
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export KIND=dq ARTIFACT=${ARTIFACT:-reuse_qdo}
export DQ_ARTIFACT=$ARTIFACT
for L in 64 128; do
  for MASK in none mixed one_key all_masked; do
    timeout 180s bash "$ENV" python -u -B "$B/dq/check.py" --length "$L" --mask "$MASK" --no-bench --output "$R/fp64-${ARTIFACT}-${L}-${MASK}-${SLURM_JOB_ID}.json"
  done
done
for L in ${LENGTHS:-384 768 1024}; do
  LN_MODULE_LENGTH=$L timeout 420s bash "$ENV" python -u -B "$R/check_module.py"
  timeout 240s bash "$ENV" python -u -B "$R/changed_graph.py" --artifact "$ARTIFACT" --length "$L" --output "$R/graph-${ARTIFACT}-${L}-${SLURM_JOB_ID}.json"
done
for L in ${SAN_LENGTHS:-64 384 768 1024}; do
  for TOOL in memcheck racecheck synccheck; do
    for MASK in mixed all_masked; do
      timeout 600s bash "$ENV" compute-sanitizer --tool "$TOOL" --kernel-name 'regex=dq_tma_reuse_qdo' --error-exitcode 86 python -u -B "$R/check_native.py" --kind dq --artifact "$ARTIFACT" --length "$L" --mask "$MASK" --native-only --output "$R/${TOOL}-${ARTIFACT}-${L}-${MASK}-${SLURM_JOB_ID}.json"
    done
  done
done
