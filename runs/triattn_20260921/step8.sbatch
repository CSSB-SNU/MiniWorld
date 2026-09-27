#!/bin/bash
#SBATCH -J mw-ta-core-ncu
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step8-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
NCU=/usr/local/cuda-12.9/bin/ncu
mkdir -p $R/ncu
# pipe mix + warp state, core kernel only
M="sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_xu.sum"
M="$M,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed.sum"
M="$M,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"
M="$M,smsp__inst_executed.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active"
M="$M,smsp__warps_eligible.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active"
M="$M,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
M="$M,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks"
M="$M,launch__registers_per_thread,launch__shared_mem_per_block_driver,gpu__time_duration.sum"
for k in "smsp__warp_issue_stalled_%s_per_warp_active.pct"; do
  for s in mio_throttle wait long_scoreboard short_scoreboard barrier no_instruction lg_throttle math_pipe_throttle imc_miss dispatch_stall membar; do
    M="$M,$(printf $k $s)"
  done
done
for L in 384 768; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; $NCU --target-processes all \
    --profile-from-start off --metrics $M --force-overwrite --export $R/ncu/core-mix-L$L \
    python -u -B $R/bench_triattn.py --length $L --row block:triattn_native --profile-one --output /dev/null" \
    > $R/logs/ncu-coremix-L$L.log 2>&1
  $NCU --import $R/ncu/core-mix-L$L.ncu-rep --page raw --csv > $R/ncu/core-mix-L$L.csv 2>/dev/null
  echo "-- L$L $(grep -c . $R/ncu/core-mix-L$L.csv) lines"
done
# source-level stalls on the core kernel alone
for L in 768 384; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; $NCU --target-processes all \
    --profile-from-start off --section SourceCounters --section WarpStateStats --section ComputeWorkloadAnalysis \
    --section InstructionStats --kernel-name regex:'(triattn_m1_kernel|_fwd)' --force-overwrite \
    --export $R/ncu/core-src-L$L \
    python -u -B $R/bench_triattn.py --length $L --row block:triattn_native --profile-one --output /dev/null" \
    > $R/logs/ncu-coresrc-L$L.log 2>&1
  echo "-- src L$L exported"
done
echo DONE
