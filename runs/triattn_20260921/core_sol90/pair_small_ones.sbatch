#!/bin/bash
#SBATCH -J mw-ta-small-ones-pair
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:15:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/small-ones-pair-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python - "$R" <<'PY'
from pathlib import Path
import hashlib,re,sys
r=Path(sys.argv[1]);s=(r/'build-basef40smallones.log').read_text()
# This older builder did not print the newer screen's BUILT marker. Verify
# its linker completion, real ELF artifact, and identical resource conditions.
assert s.rstrip().endswith('-o triattn_sol_basef40smallones.so')
so=r/'build/triattn_sol_basef40smallones/triattn_sol_basef40smallones.so'
assert so.read_bytes()[:4]==b'\x7fELF'
assert not re.search(r'C751[245]|C7520',s)
m=re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and [int(x) for x in m.groups()[1:]]==[0,0,0,128]
installed=r.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
print('PASS legacy build completion and hot resource screen; artifact SHA',hashlib.sha256(so.read_bytes()).hexdigest(),flush=True)
PY
for direction in 0 1; do
    SOL_VARIANT=basef40smallones ENDING="$direction" timeout --signal=TERM --kill-after=10s 180s bash "$ENV" python -u -B "$R/pair.py" --length 768 --output "$R/basef40smallones-paired-currentL768-${direction}.json"
done
