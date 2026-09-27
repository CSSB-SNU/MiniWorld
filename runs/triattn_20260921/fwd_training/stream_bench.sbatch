#!/bin/bash
#SBATCH -J mw-stream-bench
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/stream-bench-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" cuobjdump --dump-sass "$R/$ARTIFACT/build/triattn_training_${ARTIFACT}.so" > "$R/$ARTIFACT/sass.txt"
bash "$ENV" cuobjdump --dump-resource-usage "$R/$ARTIFACT/build/triattn_training_${ARTIFACT}.so" > "$R/$ARTIFACT/resources.txt"
bash "$ENV" python -B "$R/audit_stream_registers.py" --artifact "$ARTIFACT"
for L in 64 128 384 768 1024; do
    timeout 180s bash "$ENV" python -u -B "$R/stream_check.py" --artifact "$ARTIFACT" --length "$L" --output "$R/stream-core-${SLURM_JOB_ID}-L${L}.json"
done
