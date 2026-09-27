#!/bin/bash
#SBATCH -J mw-ta-q1024-qualify
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/qualify-q1024-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
export SOL_VARIANT=baseqthree1024
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${SOL_VARIANT}.log"
for direction in 0 1; do
 ENDING="$direction" timeout --signal=TERM --kill-after=10s 240s bash "$ENV" python -u -B "$R/pair.py" --length 1024 --output "$R/${SOL_VARIANT}-paired-L1024-${direction}.json"
done
bash "$ENV" python - "$R" "$SOL_VARIANT" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);v=sys.argv[2]
for direction in (0,1):
 d=json.loads((r/(v+'-paired-L1024-'+str(direction)+'.json')).read_text())
 assert d['ending']==bool(direction) and d['length']==1024
 assert d['summary']['core']['paired_ratio']<.9975,d['summary']
 assert d['summary']['block']['paired_ratio']<1.002,d['summary']
print('PASS paired L1024 performance screen',flush=True)
PY
SOL_FULLCASE=1 SOL_FULL_LENGTH=1024 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=0 bash "$ENV" python -u -B "$R/check_base.py"
SOL_FULLCASE=1 SOL_FULL_LENGTH=1024 CHECK_SCALE=0.25 CHECK_PATTERNS=forced_safe bash "$ENV" python -u -B "$R/check_base.py"
