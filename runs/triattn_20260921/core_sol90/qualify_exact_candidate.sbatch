#!/bin/bash
#SBATCH -J mw-ta-qualify-exact
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/qualify-exact-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
: "${SOL_VARIANT:?candidate required}"
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${SOL_VARIANT}.log"
for direction in 0 1; do
    ENDING="$direction" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/pair.py" --length 768 --output "$R/${SOL_VARIANT}-paired-currentL768-${direction}.json"
done
bash "$ENV" python - "$R" "$SOL_VARIANT" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);v=sys.argv[2]
for direction in (0,1):
    d=json.loads((r/(v+'-paired-currentL768-'+str(direction)+'.json')).read_text())
    assert d['ending']==bool(direction) and d['length']==768
    assert d['summary']['core']['paired_ratio']<.9975,d['summary']
    assert d['summary']['block']['paired_ratio']<1.002,d['summary']
print('PASS paired performance screen',flush=True)
PY
SOL_FULLCASE=1 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=0 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=1 CHECK_SCALE=0.25 CHECK_PATTERNS=forced_safe bash "$ENV" python -u -B "$R/check_base.py"
for tool in racecheck synccheck memcheck; do
    SOL_FULLCASE=1 SOL_SANITIZER=1 bash "$ENV" /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --kernel-name "kns=ta_sol_${SOL_VARIANT}" --error-exitcode 1 python -u -B "$R/check_base.py"
done
