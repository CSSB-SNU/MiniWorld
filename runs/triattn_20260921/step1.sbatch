#!/bin/bash
#SBATCH -J mw-ta-profile
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step1-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
NCU=/usr/local/cuda-12.9/bin/ncu
mkdir -p $R/ncu
# 1. per-kernel breakdown + whole-op time, the block row and (for scale) the engine's own path
for L in 384 768; do
  G=$(( L == 768 ? 0 : 1 ))
  (
    for ROW in block:triattn_native triattn_native k2b; do
      TAG=$(echo $ROW | tr ':' '-')
      bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G; python -u -B $R/bench_triattn.py --length $L --row $ROW \
        --output $R/logs/ta-$TAG-L$L.json" > $R/logs/ta-$TAG-L$L.log 2>&1
      echo "== $ROW L$L"; grep -hE "^RESULT|Traceback|Error" $R/logs/ta-$TAG-L$L.log | cut -c1-260
    done
    bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=$G; python -u -B $R/bench_triattn.py --length $L --impl miniworld \
      --output $R/logs/ta-engine-L$L.json" > $R/logs/ta-engine-L$L.log 2>&1
    echo "== engine L$L"; grep -hE "^RESULT|Traceback" $R/logs/ta-engine-L$L.log | cut -c1-200
  ) &
done
wait
# 2. NCU on the block row, same sections as the adoption campaign
for L in 384 768; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0; $NCU --target-processes all --profile-from-start off \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart --section MemoryWorkloadAnalysis \
    --section SpeedOfLight_HierarchicalTensorRooflineChart --section SchedulerStats \
    --section LaunchStats --section Occupancy --force-overwrite --export $R/ncu/ta-block-L$L \
    python -u -B $R/bench_triattn.py --length $L --row block:triattn_native --profile-one --output /dev/null" \
    > $R/logs/ncu-block-L$L.log 2>&1
  $NCU --import $R/ncu/ta-block-L$L.ncu-rep --page raw --csv > $R/ncu/ta-block-L$L.csv 2>/dev/null
  echo "-- ncu L$L: $(grep -c . $R/ncu/ta-block-L$L.csv 2>/dev/null) csv lines"
done
echo DONE
