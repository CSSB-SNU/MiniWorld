#!/bin/bash
#SBATCH -J mw-ta-lay
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step25-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
E="CUDA_VISIBLE_DEVICES=0 CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 TORCH_EXTENSIONS_DIR=$R/ext-flags MAX_JOBS=16"
# Does the core care WHICH physical tensor carries the [B,N,H,S,D] it consumes?  If it does not, the prologue can write
# q/k/v in pair-natural order (256 B runs, fully coalesced) instead of the [I,H,J,D] scatter (64 B runs) it writes today.
for L in 768 384 1024; do
  for LAY in nhsd nshd shnd; do
    bash $ENV bash -c "export $E; python -u -B $R/bench_core_flags.py --length $L --flags 0 --qkv-layout $LAY \
      --output $R/logs/lay-$LAY-L$L.json" > $R/logs/lay-$LAY-L$L.log 2>&1
    echo "===== L$L qkv=$LAY"; grep -E "^flags|strides" $R/logs/lay-$LAY-L$L.log || tail -12 $R/logs/lay-$LAY-L$L.log
  done
done
python3 - <<'PY'
import json, os
R="/home/psk6950/MiniWorld/runs/triattn_20260921"
print("%-6s %-8s %10s %9s  %s"%("L","qkv","core us","vs nhsd","max|err|"))
for L in (384,768,1024):
    base=None
    for lay in ("nhsd","nshd","shnd"):
        p=f"{R}/logs/lay-{lay}-L{L}.json"
        if not os.path.exists(p): continue
        r=json.load(open(p))["rows"]
        if not r: continue
        us=r[0]["us"]; base=base or us
        print("%-6d %-8s %10.1f %8.1f%%  %.3e"%(L,lay,us,100*(us/base-1),r[0]["max_abs"]))
PY
echo DONE
