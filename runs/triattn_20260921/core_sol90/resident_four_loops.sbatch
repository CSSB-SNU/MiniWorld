#!/bin/bash
#SBATCH -J mw-ta-resident-loop
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/resident-four-loops-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
for kind in loop6 two8; do
 if [ "$kind" = loop6 ]; then VARIANT=resident4m64s3qrloop6; else VARIANT=resident4m64s2qrloop8; fi
 bash "$ENV" python -u "$R/build_resident_four_m64.py" --kind="$kind" > "$R/build-${VARIANT}.log" 2>&1
 if ! bash "$ENV" python "$R/screen_resident_four.py" "$R/build-${VARIANT}.log"; then
  echo "REJECTED BEFORE GPU $VARIANT"
  continue
 fi
 SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
done
