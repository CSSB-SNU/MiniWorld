#!/bin/bash
#SBATCH -J mw-stream-build
#SBATCH -p h100
#SBATCH --nodelist=node02
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training/stream-build-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/fwd_training
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
KEEP_PTX=1 bash "$ENV" python -u -B "$R/native.py" --artifact "$ARTIFACT"
bash "$ENV" cuobjdump --dump-sass "$R/$ARTIFACT/build/triattn_training_${ARTIFACT}.so" > "$R/$ARTIFACT/sass.txt"
bash "$ENV" cuobjdump --dump-resource-usage "$R/$ARTIFACT/build/triattn_training_${ARTIFACT}.so" > "$R/$ARTIFACT/resources.txt"
bash "$ENV" python -B "$R/audit_stream_registers.py" --artifact "$ARTIFACT"
timeout 120s bash "$ENV" python -u -B "$R/stream_check.py" --artifact "$ARTIFACT" --length 64 --output "$R/stream-pilot-${SLURM_JOB_ID}.json"
