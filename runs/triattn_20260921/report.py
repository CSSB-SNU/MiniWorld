"""NCU + CUPTI table for the Anthropic TriangleAttention block."""
import csv, json
M = {"dur": "gpu__time_duration.sum", "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
     "tensor": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
     "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
     "l2": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
     "reg": "launch__registers_per_thread", "grid": "launch__grid_size", "block": "launch__block_size",
     "smem": "launch__shared_mem_per_block_driver"}
SHORT = {"_triatt_prologue_kernel": "prologue (fpf v3)", "_triatt_epilogue_kernel_v2": "epilogue (fpf v2)",
         "_fwd": "core: Triton _fwd", "_stage_bias": "stage_bias"}
def short(n):
    for k, v in SHORT.items():
        if n.startswith(k):
            return v
    if "triattn_m1_kernel" in n:
        return "core: CUDA triattn_m1"
    if "vectorized_elementwise" in n:
        return "residual add (torch)"
    for t in ("mask_words_kernel", "mask_rows_kernel", "stage_bias_m1_kernel", "uniform_rows_kernel"):
        if t in n:
            return t.replace("_kernel", "")
    return n[:34]
for L in (384, 768):
    cu = json.load(open("logs/ta-block-triattn_native-L%d.json" % L))
    cupti = {short(k["name"]): k["us"] for k in cu["kernels"]}
    f = open("ncu/ta-block-L%d.csv" % L); rd = csv.reader(f); hdr = next(rd); units = dict(zip(hdr, next(rd)))
    rows = [dict(zip(hdr, r)) for r in rd if r and r[0].strip().isdigit()]
    scale = 1000.0 if units[M["dur"]] == "ms" else 1.0
    print("\n== L%d   whole op %.1f us (one-call CUDA graph), rel_rms %.3e" % (L, cu["op_us"], cu["error"]["rel_rms"]))
    print("   %-22s %8s %7s %6s %6s %7s %6s %6s %5s %10s" % ("kernel", "CUPTI us", "% op", "SM%", "DRAM%", "tensor%", "occ%", "L2%", "regs", "grid"))
    tot = 0.0
    for r in sorted(rows, key=lambda r: -float(r[M["dur"]] or 0)):
        n = short(r["Kernel Name"]); us = cupti.get(n, float("nan")); tot += us if us == us else 0
        g = lambda k: float(r[M[k]]) if r.get(M[k]) not in (None, "", "n/a") else float("nan")
        print("   %-22s %8.1f %6.1f%% %6.1f %6.1f %7.1f %6.1f %6.1f %5s %10s"
              % (n, us, 100 * us / cu["op_us"], g("sm"), g("dram"), g("tensor"), g("occ"), g("l2"),
                 r.get(M["reg"], "?"), r.get(M["grid"], "?")))
    print("   %-22s %8.1f %6.1f%%   (the rest is launch gap)" % ("kernels total", tot, 100 * tot / cu["op_us"]))
