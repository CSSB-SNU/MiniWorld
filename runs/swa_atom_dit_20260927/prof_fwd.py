import os, sys, torch
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_fwd.py").read().split("with torch.no_grad():")[0])
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    m, q, c1, c, cos, sin, valid, ap = make(48, 1, 4096)
    for _ in range(3): ours(m, q, c1, cos, sin, valid, 1)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3): ours(m, q, c1, cos, sin, valid, 1)
        torch.cuda.synchronize()
rows = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
for e in rows[:8]: print("  %8.1f us  %s" % (e.self_device_time_total / 3, e.key[:80]))
M = 48 * 4096
for k in ("qkvg", "attn", "ffn"):
    kk = TS.LAST[k]; print(k, "regs", kk.n_regs, "spills", kk.n_spills)
