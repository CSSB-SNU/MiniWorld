#!/bin/bash
#SBATCH -J mw-ta-ncu2
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step4-%j.out
set -x
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
NCU=/usr/local/cuda-12.9/bin/ncu
M="dram__bytes_read.sum,dram__bytes_write.sum"
M="$M,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
M="$M,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
M="$M,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
M="$M,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_wait_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_barrier_per_warp_active.pct"
M="$M,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"
M="$M,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps"
M="$M,launch__registers_per_thread,launch__grid_size,launch__block_size,gpu__time_duration.sum"
M="$M,sm__sass_average_data_bytes_per_sector_mem_global_op_st.pct,sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
for L in 384 768; do
  bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH; $NCU --target-processes all \
    --profile-from-start off --metrics $M --force-overwrite --export $R/ncu/ta-mem-L$L --csv --page raw \
    python -u -B $R/bench_triattn.py --length $L --row block:triattn_native --profile-one --output /dev/null" \
    > $R/ncu/ta-mem-L$L.csv 2> $R/logs/ncu-mem-L$L.log
  echo "-- L$L: $(grep -c . $R/ncu/ta-mem-L$L.csv) csv lines"; tail -2 $R/logs/ncu-mem-L$L.log | cut -c1-140
done
echo DONE
