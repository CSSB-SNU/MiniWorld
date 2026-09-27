"""team_gm SWAAtomTransformer: engine path vs fused Triton path (fused_triton=True), with checkpointing, c_base given / not given."""
import os, sys, copy, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_fwd.py").read().split("def ours(")[0])
from team_gm.modules import SWAAtomTransformer
def rel(a, b): a = a.float(); b = b.float(); return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
def t(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s0 = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s0.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s0.elapsed_time(e) / it
for A, B, S, su, ckpt in ():
    m, q, c1, c, cos, sin, valid, ap = make(A, B, S, su)
    m.config.n_checkpoint_segments = ckpt
    mf = copy.deepcopy(m); mf.config.fused_triton = True
    m32 = copy.deepcopy(m).float()
    dy = torch.randn_like(q) * valid[..., None]
    q32 = q.float().requires_grad_(True); c132 = c1.float().requires_grad_(True)
    r = ref_fp32(m32, q32, c132.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, S, -1), cos, sin, valid, A)
    gr = torch.autograd.grad(r, [q32, c132] + list(m32.parameters()), dy.float())
    res = {}
    for name, mod_, use_base in (("engine", m, False), ("fused+c_base", mf, True), ("fused no c_base", mf, False)):
        qq = q.clone().requires_grad_(True); cc = c1.clone().requires_grad_(True)
        cexp = cc.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, S, -1)
        out = mod_(qq, cexp, ap, cc if use_base else None)
        g = torch.autograd.grad(out, [qq, cc] + list(mod_.parameters()), dy)
        errs = [rel(out[valid], r[valid])] + [rel(a[valid] if i == 0 else a, (b[valid] if i == 0 else b)) for i, (a, b) in enumerate(zip(g, gr))]
        res[name] = errs
    print("A=%d B=%d S=%d seqused=%s ckpt=%s | out / worst grad vs fp32: %s" % (A, B, S, su, ckpt,
          "  ".join("%s %.1e / %.1e" % (k, v[0], max(v[1:])) for k, v in res.items())), flush=True)
# speed through the module (checkpointing as in the diffusion config)
for A, S, train in ((5, 4096, False), (48, 4096, True)):
    m, q, c1, c, cos, sin, valid, ap = make(A, 1, S); m.config.n_checkpoint_segments = 3
    mf = copy.deepcopy(m); mf.config.fused_triton = True
    P = list(m.parameters()); Pf = list(mf.parameters())
    if train:
        qq = q.clone().requires_grad_(True); cc = c1.clone().requires_grad_(True); dy = torch.randn_like(q)
        fe = lambda: torch.autograd.grad(m(qq, cc.unsqueeze(0).expand(A, -1, -1, -1).reshape(A, S, -1), ap), [qq, cc] + P, dy)
        ff = lambda: torch.autograd.grad(mf(qq, cc.unsqueeze(0).expand(A, -1, -1, -1).reshape(A, S, -1), ap, cc), [qq, cc] + Pf, dy)
        te, tf = t(fe, 5), t(ff, 5); label = "train fwd+bwd (ckpt 3)"
    else:
        with torch.no_grad():
            te = t(lambda: m(q, c, ap)); tf = t(lambda: mf(q, c, ap, c1)); label = "inference fwd"
    print("A=%d S=%d %s: engine %.3f ms | fused %.3f ms (%.2fx)" % (A, S, label, te, tf, te / tf), flush=True)
