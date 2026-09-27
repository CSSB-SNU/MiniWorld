#!/bin/bash
#SBATCH -J mw-ta-ex2v
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step15-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
V=$R/oc/opt_core/kernels/triattn/triattn_native/pkg/v11
TV=$V/testvectors

# ---- 1. accuracy of the f16x2 core against the package's own fp64 reference (pristine manifest = the stock kernel's recorded error).
#         --impl pkg imports triattn_pkg directly, so the byte gate does not stand between us and the measurement.
python3 $R/install_variant.py pb-f16x2b
bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$V:\$PYTHONPATH; \
  python -u -B $V/test_pkg.py --impl pkg --json $R/logs/acc-f16x2b.json" > $R/logs/acc-f16x2b.log 2>&1
tail -25 $R/logs/acc-f16x2b.log

# ---- 2. the variant's own byte vectors (it cannot reproduce the stock bytes by construction; step 1 is what says it is correct)
bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$V:\$PYTHONPATH; \
  python -u -B $V/test_pkg.py --impl pkg --generate --note 'TRIATTN_EX2_F16X2 variant of triattn_m1_ext' " > $R/logs/gen-f16x2b.log 2>&1
tail -22 $R/logs/gen-f16x2b.log
mkdir -p $R/tv/pb-f16x2b
cp $TV/manifest.json $R/tv/pb-f16x2b/
for f in $TV/*.expected.pt; do cp "$f" $R/tv/pb-f16x2b/; done
# which cases actually changed bytes: only the cuda_b route should have
python3 - <<'PY'
import hashlib, os, glob
R="/home/psk6950/MiniWorld/runs/triattn_20260921"
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
for f in sorted(glob.glob(R+"/tv/pristine/*.expected.pt")):
    n=os.path.basename(f); g=R+"/tv/pb-f16x2b/"+n
    if os.path.exists(g) and sha(f)!=sha(g): print("bytes changed:", n)
PY

# ---- 3. the A/B, each variant byte-gated against its own vectors
for RD in r1 r2 r3; do
for T in base2 f16x2b; do
  python3 $R/install_variant.py pb-$T
  for L in 384 768; do
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; \
      python -u -B $R/sweep_pro.py --length $L --output $R/logs/x5-$T-L$L-$RD.json" > $R/logs/x5-$T-L$L-$RD.log 2>&1
    if grep -q "refused by kernels.triattn" $R/logs/x5-$T-L$L-$RD.log; then
      echo "!! $T L$L $RD FELL BACK: $(grep -oE 'refused by kernels.triattn \([^)]*\)' $R/logs/x5-$T-L$L-$RD.log | head -1 | cut -c1-140)"
    fi
  done
done
done
python3 $R/summarise_ab.py x5 base2 f16x2b || true
echo DONE
