#!/bin/bash
#SBATCH -J mw-ta-coop-follow
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:25:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/coop-three-follow-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
VARIANT=m128r4coop3p2ip4v2${COOP_MODE:?}
bash "$ENV" python -u "$R/build_coop_three_score.py" "$COOP_MODE" > "$R/build-${VARIANT}.log" 2>&1
bash "$ENV" python - "$R/build-${VARIANT}.log" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert 'BUILT m128r4coop3p2ip4v2' in s
assert not re.search(r'C751[245]|C7520',s)
found=0
for chunk in re.split(r'(?=ptxas info\s+: Compiling entry function)',s):
    if 'attentionILb' not in chunk.splitlines()[0]:continue
    found+=1
    m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',chunk)
    assert m and list(map(int,m.groups()[:3]))==[0,0,0] and int(m[4])<=128,m.groups() if m else None
assert found==2
print('PASS cooperative3-score resource screen')
PY
SOL_VARIANT="$VARIANT" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/${VARIANT}-L768.json"
