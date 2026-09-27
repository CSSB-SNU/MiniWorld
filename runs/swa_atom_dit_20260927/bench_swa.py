import os, sys, torch
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_bwd.py").read().split("for A, B, S, su in")[0])
from torch.profiler import profile, ProfilerActivity
for A, S in ((5, 4096), (48, 4096)):
    m, q, c1, c, cos, sin, valid, ap = make(A, 1, S)
    P = params(m); dy = torch.randn_like(q)
    qe = q.clone().requires_grad_(True); ce = c1.clone().requires_grad_(True)
    fe = lambda: torch.autograd.grad(m(qe, ce.unsqueeze(0).expand(A, -1, -1, -1).reshape(A, S, -1), ap), [qe, ce] + P, dy)
    fo = lambda: torch.autograd.grad(ours_diff(m, qe, ce, cos, sin, valid, 1), [qe, ce] + P, dy)
    te, to = t(fe, 5), t(fo, 5)
    print("A=%d S=%d fwd+bwd (3 blocks): engine %.3f ms | ours %.3f ms (%.2fx)" % (A, S, te, to, te / to), flush=True)
    if A == 48:
        for _ in range(2): fo()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3): fo()
            torch.cuda.synchronize()
        rows = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in rows)
        for e in rows[:16]: print("  %8.1f us %5.1f%%  %s" % (e.self_device_time_total / 3, 100 * e.self_device_time_total / tot, e.key[:80]))
        print("  %8.1f us total" % (tot / 3))
        for k, v in TS.LAST.items(): print("  %-9s regs %3d spills %4d" % (k, v.n_regs, v.n_spills))
        for kn in ("_ffn_bwd", "_ffn_dw", "_oproj_bwd", "_qkvg_bwd", "_oproj_ffn_fwd", "_qkvg_fwd"):
            kk = getattr(TS, kn, None)
            if kk is not None and getattr(kk, "best_config", None) is not None: print("  best", kn, kk.best_config)
