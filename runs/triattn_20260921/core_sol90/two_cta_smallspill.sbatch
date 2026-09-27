#!/bin/bash
#SBATCH -J mw-ta-twocta-probe
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:10:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/twocta-probe-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python - "$R" <<'PY'
from pathlib import Path
import re,sys
r=Path(sys.argv[1]);s=(r/'build-m128r2s1p1twoctafsc.log').read_text()
assert 'BUILT m128r2s1p1twoctafsc' in s and not re.search(r'C751[245]|C7520',s)
m=re.search(r'Function properties for (\S*attentionILb0\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and [int(x) for x in m.groups()[1:]]==[16,12,12,96]
print('BOUNDED PROBE:96regs,16B stack,12B stores/12B loads; NOT spill-free. SASS has3STL/3LDL static sites. Runtime checks two-CTA residency.',flush=True)
PY
SOL_VARIANT=m128r2s1p1twoctafsc timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/bench.py" --length 768 --output "$R/m128r2s1p1twoctafsc-L768.json"
