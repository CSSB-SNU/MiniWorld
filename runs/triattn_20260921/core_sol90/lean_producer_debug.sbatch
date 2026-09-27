#!/bin/bash
#SBATCH -J mw-ta-lean-debug
#SBATCH -p h100
#SBATCH --qos=cssb_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:03:00
#SBATCH -o /home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90/lean-producer-debug-%j.log
set -euo pipefail
R=/home/psk6950/MiniWorld/runs/triattn_20260921/core_sol90
ENV=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh
BENCH_LOG="$R/lean-producer-debug-${SLURM_JOB_ID}-bench.log"
export CUBIN_CASES="$R/lean-producer-b8-cases.json" CUBIN_DEBUG_STAGE=1 ENDING=0
timeout --signal=TERM --kill-after=5s 100s bash "$ENV" python -u -B "$R/bench_ptxas_graph.py" --length 768 --output "$R/lean-producer-debug-${SLURM_JOB_ID}.json" > "$BENCH_LOG" 2>&1 &
runner_pid=$!
trap 'kill "$runner_pid" 2>/dev/null || true' EXIT
for probe in $(seq 1 30); do
    sleep 2
    if ! kill -0 "$runner_pid" 2>/dev/null; then break; fi
    if rg -q '^STAGE core lean24q168b8 synchronize' "$BENCH_LOG" && ! rg -q '^BITWISE core lean24q168b8' "$BENCH_LOG"; then
        bench_pid=$(pgrep -f "^python -u -B $R/bench_ptxas_graph.py --length 768 --output $R/lean-producer-debug-${SLURM_JOB_ID}.json$")
        printf 'ATTACH_OWN_BENCH_PID %s\n' "$bench_pid"
        timeout --signal=TERM --kill-after=5s 25s /usr/local/cuda-12.9/bin/cuda-gdb -batch -p "$bench_pid" \
          -ex 'set pagination off' -ex 'info cuda kernels' -ex 'info cuda sms' \
          -ex 'cuda device 0 sm 0' -ex 'info cuda warps' -ex 'x/12i $pc' -ex 'bt 4' -ex 'detach' || true
        break
    fi
done
set +e
wait "$runner_pid"
status=$?
set -e
printf 'BENCH_EXIT %s\n' "$status"
tail -12 "$BENCH_LOG"
exit "$status"
