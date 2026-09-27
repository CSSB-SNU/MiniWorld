#!/bin/bash
#SBATCH -J mw-ta-ncu2
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/logs/step28-%j.out
set -x
echo "NODE=$(hostname)"
R=/home/psk6950/MiniWorld/runs/triattn_20260921
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
BEST="FPF_TRIATTPRO_BN=64 FPF_TRIATTPRO_MAXNREG=128 FPF_TRIATT_EPI_CFG=8,8,4,1"
# where the two memory kernels stand NOW (best settings), with the bytes and the roofs in one pass
bash $ENV bash -c "export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/oc:\$PYTHONPATH $BEST; \
  /usr/local/cuda/bin/ncu --target-processes all -k regex:'triatt_prologue|triatt_epilogue' -c 4 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SourceCounters \
    -o $R/ncu/best-L768 --force-overwrite \
    python -u -B $R/sweep_pro.py --length 768 --iters 2 --output $R/logs/ncu-best-L768.json" > $R/logs/ncu-best-L768.log 2>&1
tail -3 $R/logs/ncu-best-L768.log
/usr/local/cuda/bin/ncu --import $R/ncu/best-L768.ncu-rep --page details --section SpeedOfLight 2>/dev/null | head -40
echo "---- memory + occupancy + stalls"
/usr/local/cuda/bin/ncu --import $R/ncu/best-L768.ncu-rep --page raw --csv 2>/dev/null > $R/logs/best-raw.csv
python3 - <<'PY'
import csv
R="/home/psk6950/MiniWorld/runs/triattn_20260921"
want=("dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum","launch__registers_per_thread",
      "launch__occupancy_limit_registers","sm__warps_active.avg.pct_of_peak_sustained_active",
      "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
      "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
      "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
      "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
      "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
      "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
      "smsp__warp_issue_stalled_wait_per_warp_active.pct",
      "smsp__issue_active.avg.pct_of_peak_sustained_active")
seen=set()
for d in csv.DictReader(open(R+"/logs/best-raw.csv")):
    k=d.get("Kernel Name","")
    if k in seen or not k: continue
    seen.add(k); print("###",k[:50],"grid",d.get("Grid Size"),"blk",d.get("Block Size"))
    try:
        t=float(d["gpu__time_duration.sum"])*1e-3; b=float(d["dram__bytes_read.sum"])+float(d["dram__bytes_write.sum"])
        print("    %.1f us  DRAM %.1f MB -> %.2f TB/s (%.0f%% of the 2.95 TB/s measured stream)"%(t*1e6,b/1e6,b/t/1e12,100*b/t/1e12/2.95))
    except Exception as e: print("   ",e)
    for c in want:
        v=d.get(c)
        if v not in (None,""): print("    %-66s %s"%(c[:66],v))
PY
echo DONE
