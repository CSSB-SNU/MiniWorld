#!/bin/bash
#SBATCH -J mw-ta-qthree1024
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/qthree1024-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=baseqthree1024
bash "$ENV" python -u "$R/build_specialize_1024_q.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python "$R/screen_installed_experiment.py" "$R/build-${VARIANT}.log"
bash "$ENV" python - "$R/build-${VARIANT}.log" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
m=re.search(r'Function properties for (\S*536870912\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and list(map(int,m.groups()[1:]))==[0,0,0,128],m.groups() if m else None
print('PASS L1024 hot resource screen',flush=True)
PY
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 1024 --output "$R/${VARIANT}-L1024.json"
