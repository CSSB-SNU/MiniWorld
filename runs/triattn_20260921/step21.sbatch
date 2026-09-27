#!/bin/bash
#SBATCH -J mw-ta-bias
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step21-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
E="CUDA_VISIBLE_DEVICES=0 CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 TORCH_EXTENSIONS_DIR=$R/ext-flags MAX_JOBS=16"
# The pair bias is [1,1,H,S,S] and is re-read by every one of the N pair rows, so it is the kernel's dominant L2 traffic.
# Its VALUES are a bf16 GEMM output either way, so handing it as bf16 halves those bytes and changes no arithmetic.
for L in 768 384; do
  for BD in fp32 bf16; do
    bash $ENV bash -c "export $E; python -u -B $R/bench_core_flags.py --length $L --flags 0,8 --bias-dtype $BD \
      --output $R/logs/bias-$BD-L$L.json" > $R/logs/bias-$BD-L$L.log 2>&1
    echo "===== L$L bias=$BD"; grep -E "^flags" $R/logs/bias-$BD-L$L.log || tail -20 $R/logs/bias-$BD-L$L.log
  done
done
# where the core's bytes actually go: DRAM vs L2, fp32 bias vs bf16 bias
for BD in fp32 bf16; do
  bash $ENV bash -c "export $E; /usr/local/cuda/bin/ncu --target-processes all -k regex:triattn_m1_kernel -c 2 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,gpu__time_duration.sum \
    -o $R/ncu/corebias-$BD --force-overwrite \
    python -u -B $R/bench_core_flags.py --length 768 --flags 0 --bias-dtype $BD --iters 2 --warmup 1" > $R/logs/ncu-bias-$BD.log 2>&1
  echo "===== ncu bias=$BD"
  /usr/local/cuda/bin/ncu --import $R/ncu/corebias-$BD.ncu-rep --page raw --csv 2>/dev/null | python3 -c "
import csv,sys
for d in csv.DictReader(sys.stdin):
    if 'triattn_m1_kernel' not in d.get('Kernel Name',''): continue
    t=float(d['gpu__time_duration.sum'])*1e-3
    dr=float(d['dram__bytes_read.sum'])+float(d['dram__bytes_write.sum'])
    l2=(float(d['lts__t_sectors_op_read.sum'])+float(d['lts__t_sectors_op_write.sum']))*32
    print('  %7.3f ms  DRAM %7.1f MB (%5.2f TB/s)  L2 %8.1f MB (%5.2f TB/s)'%(t*1e3,dr/1e6,dr/t/1e12,l2/1e6,l2/t/1e12))
    break
"
done
echo DONE
