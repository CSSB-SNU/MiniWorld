"""Current MiniWorld ESMFold2 SWA atom DiT (3 blocks, bf16): timing + per-kernel profile."""
import os, sys, torch
from torch.profiler import profile, ProfilerActivity
from team_gm.modules import SWAAtomTransformer
from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
dev = "cuda"; d = 128
def make(N_aug, B, S, seed=0):
    torch.manual_seed(seed)
    cfg = SWAAtomTransformer.Config(d_atom=d, d_cond=d, n_block=3, n_head=4, swa_window_size=128, expansion_ratio=2)
    m = SWAAtomTransformer(cfg).to(dev).to(torch.bfloat16)
    for blk in m.blocks:                                     # adaLN-Zero init is zero: randomize so every path is exercised
        torch.nn.init.normal_(blk.adaln_modulation[1].weight, std=0.02)
    ref_pos = torch.randn(B, S, 3, device=dev) * 5; uid = torch.arange(S, device=dev).view(1, S).expand(B, S) // 8
    cos, sin = m.build_rope(ref_pos, uid)
    valid = torch.ones(N_aug * B, S, dtype=torch.bool, device=dev)
    ap = build_attention_params(cos, sin, valid, N_aug)
    q = torch.randn(N_aug * B, S, d, device=dev, dtype=torch.bfloat16)
    c1 = torch.randn(B, S, d, device=dev, dtype=torch.bfloat16)
    c = c1.unsqueeze(0).expand(N_aug, -1, -1, -1).reshape(N_aug * B, S, d)   # as the diffusion module: augment-invariant cond
    return m, q, c, ap
def t(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
for label, N, S, train in (("train N=48 S=4096", 48, 4096, True), ("infer N=5 S=4096", 5, 4096, False)):
    m, q, c, ap = make(N, 1, S)
    if train:
        qq = q.clone().requires_grad_(True); dy = torch.randn_like(q)
        fn = lambda: torch.autograd.grad(m(qq, c, ap), [qq] + list(m.parameters()), dy)
        tf = t(lambda: m(q, c, ap).sum()) 
        with torch.no_grad(): tf = t(lambda: m(q, c, ap))
        tt = t(fn)
        print("%s: fwd %.3f ms, fwd+bwd %.3f ms" % (label, tf, tt), flush=True)
    else:
        with torch.no_grad():
            fn = lambda: m(q, c, ap)
            tt = t(fn)
        print("%s: fwd %.3f ms" % (label, tt), flush=True)
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for _ in range(2): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3): fn()
            torch.cuda.synchronize()
    rows = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
    tot = sum(e.self_device_time_total for e in rows)
    for e in rows[:22]: print("  %8.1f us %5.1f%%  %s" % (e.self_device_time_total / 3, 100 * e.self_device_time_total / tot, e.key[:95]))
    print("  %8.1f us total GPU" % (tot / 3), flush=True)
