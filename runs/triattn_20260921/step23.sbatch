#!/bin/bash
#SBATCH -J mw-ta-rel
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step23-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
V=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11
CU=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2
# 1. the core source is now byte-identical to the release again (the f16x2 experiment is reverted), so ONE prebuilt serves
#    both trees.  Build it from that source and record its own loadcheck.
bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 CUTLASS_PATH=$CU MAX_JOBS=16 PYTHONPATH=$V:\$PYTHONPATH; \
  cd $V && python -u -B tools/build_prebuilt.py --exts triattn_m1_ext" > $R/logs/rebuild-pristine.log 2>&1
tail -6 $R/logs/rebuild-pristine.log
# 2. the same prebuilt into the release copy (identical source -> identical digests)
PB=$V/triattn_pkg/prebuilt
RPB=$R/oc-release/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt
rm -rf $RPB && cp -r $PB $RPB
# build_prebuilt.py writes the .so but not the two SHA256SUMS the face checks first: without this the block silently uses flash
python3 $R/refresh_sums.py $R/oc $R/oc-release
python3 - <<'PY'
import hashlib, os
R="/home/psk6950/MiniWorld/runs/triattn_20260921"
a=R+"/oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt/torch2.10.0+cu128-cpython-310-x86_64-linux-gnu/triattn_m1_ext.so"
b=R+"/oc-release/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt/torch2.10.0+cu128-cpython-310-x86_64-linux-gnu/triattn_m1_ext.so"
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print("ours .so", sha(a), "| release .so", sha(b), "| same:", sha(a)==sha(b))
PY
# 3. the block, both trees, same process shape.  Ours carries its measured settings through the env overrides until the
#    cell rows are updated, so the number is what the wired-in version will do.
OURS="FPF_TRIATTPRO_BN=64 FPF_TRIATTPRO_MAXNREG=128 FPF_TRIATT_EPI_CFG=8,8,4,1"   # epilogue: 8x8 alone; maxnreg measured WORSE with it
for L in 384 768 1024; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 CUTLASS_PATH=$CU; \
    python -u -B $R/bench_vs_release.py --oc $R/oc-release --tag release --length $L --output $R/logs/vsrel-release-L$L.json" \
    > $R/logs/vsrel-release-L$L.log 2>&1
  tail -4 $R/logs/vsrel-release-L$L.log | grep -E "release |Error|error" | head -3
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 CUTLASS_PATH=$CU $OURS; \
    python -u -B $R/bench_vs_release.py --oc $R/oc --tag ours --length $L --output $R/logs/vsrel-ours-L$L.json" \
    > $R/logs/vsrel-ours-L$L.log 2>&1
  tail -4 $R/logs/vsrel-ours-L$L.log | grep -E "ours |Error|error" | head -3
done
python3 - <<'PY'
import json, os
R="/home/psk6950/MiniWorld/runs/triattn_20260921"
print("%-6s %-11s %10s %10s %9s  %s"%("L","form","release","ours","speedup","pieces (release -> ours)"))
for L in (384,768,1024):
    try:
        a=json.load(open(f"{R}/logs/vsrel-release-L{L}.json")); b=json.load(open(f"{R}/logs/vsrel-ours-L{L}.json"))
    except OSError: continue
    for f in ("inplace","outofplace"):
        ra,rb=a.get(f,{}),b.get(f,{})
        if "op_us" not in ra or "op_us" not in rb:
            print("%-6d %-11s %s"%(L,f,ra.get("unsupported") or rb.get("unsupported") or "missing")); continue
        print("%-6d %-11s %10.1f %10.1f %8.3fx  pro %.0f->%.0f  core %.0f->%.0f  epi %.0f->%.0f  rel_rms %.3e/%.3e"
              %(L,f,ra["op_us"],rb["op_us"],ra["op_us"]/rb["op_us"],
                ra["prologue_us"],rb["prologue_us"],ra["core_us"],rb["core_us"],ra["epilogue_us"],rb["epilogue_us"],
                ra["rel_rms"],rb["rel_rms"]))
PY
echo DONE
