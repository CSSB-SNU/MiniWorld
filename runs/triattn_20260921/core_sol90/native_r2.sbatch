#!/bin/bash
#SBATCH -J mw-ta-native-r2
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/native-r2-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=basenativer2qr
if [ "${NATIVE_R2_QR:-1}" = 0 ]; then VARIANT=basenativer2ss; fi
bash "$ENV" python -u "$R/build_native_r2.py" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python - "$R/build-${VARIANT}.log" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text();assert 'BUILT basenativer2' in s
assert not re.search(r'C751[245]|C7520',s),'serialized MMA'
m=re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and [int(x) for x in m.groups()[1:4]]==[0,0,0]
assert 160<=int(m.group(5))<=168,m.groups()
print('PASS clean R2 hot with adequate initial CTA register pool',m.groups(),flush=True)
PY
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
