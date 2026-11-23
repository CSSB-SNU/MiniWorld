#!/bin/bash
#SBATCH --job-name=mw_fb_score
#SBATCH --partition=h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/psk6950/MiniWorld/runs/eval_results/foldbench/score_%j.out
#SBATCH --error=/home/psk6950/MiniWorld/runs/eval_results/foldbench/score_%j.out
#
# FoldBench scoring. CPU ONLY -- no --gres, so this never touches the GPU budget.
#
# RUN THIS AS A BATCH JOB, NEVER ON THE LOGIN NODE. upstream/evaluation/eval_by_ost.py
# fans out to max_workers=64 threads and each one launches its own OpenStructure
# singularity container, so running it interactively pins the login node and locks
# everyone out. --cpus-per-task=64 matches that fan-out to real cores.
#
# Per-prediction OST results are cached in <eval-dir>/<name>/detail/*.json and skipped
# if present, so re-running after more predictions land only scores the new ones.
#
# upstream hard-codes max_workers=64 (OST) / 32 (DockQv2); MW_OST_WORKERS and
# MW_DOCKQ_WORKERS now override that. Each worker is a separate OST process, so on a
# 224-core node the default left ~two thirds of the machine idle. Pinned to node02,
# which has 160 cores free next to the 64 the training job holds.
set -uo pipefail
cd /home/psk6950/MiniWorld

NAME="${MW_EVAL_NAME:-p2b-swa-5x5}"
PRED="${MW_PRED_DIR:-runs/foldbench/v1.0.1-phase2b-swa-5x5}"
EVAL=/home/psk6950/MiniWorld/runs/eval_results/foldbench
PY=/home/psk6950/data/foldbench/upstream/.venv/bin/python
export MW_OST_WORKERS="${MW_OST_WORKERS:-30}"
export MW_DOCKQ_WORKERS="${MW_DOCKQ_WORKERS:-16}"
mkdir -p "$EVAL"

echo "=== FoldBench scoring | job $SLURM_JOB_ID | $(hostname) ==="
echo "pred=$PRED name=$NAME  CIFs=$(find "$PRED" -name '*_pred.cif' | wc -l)"
echo "workers: ost=$MW_OST_WORKERS dockq=$MW_DOCKQ_WORKERS  cpus=$SLURM_CPUS_PER_TASK"

REQ="${MW_REQUIRE_SAMPLES:-0}"
"$PY" scripts/foldbench_evaluate.py --pred-dir "$PRED" --eval-dir "$EVAL" --name "$NAME" \
    --require-samples "$REQ"
rc=$?
echo "==== evaluate rc=$rc ===="

"$PY" scripts/foldbench_compare.py --eval-dir "$EVAL" --name "$NAME" \
    --out "$EVAL/compare_${NAME}.txt"
echo "==== EXIT $rc ===="
exit "$rc"
