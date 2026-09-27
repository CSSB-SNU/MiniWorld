#!/bin/bash
#SBATCH -J mw-ta-scaled-q-current
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:20:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/scaled-q-current-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for kind in full three; do
 VARIANT=baseqscaledcurrent${kind}
 bash "$ENV" python -u "$R/build_scaled_q_current.py" "$kind" > "$R/build-${VARIANT}.log" 2>&1
 if ! bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"; then
  echo "REJECTED BEFORE GPU $VARIANT"
  continue
 fi
 SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
done
