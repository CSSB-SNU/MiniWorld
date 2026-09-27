"""Gradients of the fused stack vs an fp32 dense reference (and the engine's gradients vs the same reference, as the baseline)."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_fwd.py").read().split("with torch.no_grad():")[0])
def ours_diff(m, q, c1, cos, sin, valid, B):
    seq = valid.sum(-1, dtype=torch.int32); S = q.shape[1]
    cs, sn = cos.reshape(B * S, -1).contiguous(), sin.reshape(B * S, -1).contiguous()
    x = q
    for blk in m.blocks:
        mod = TS.hoist_mod(c1, blk.adaln_modulation[1].weight)
        x = TS.swa_block(x, mod, cs, sn, seq, blk.attn.Wqkv.weight, blk.attn.gate_proj.weight, blk.attn.out_proj.weight, blk.ffn.w_up.weight, blk.ffn.w_down.weight, B)
    return x
def params(m): return [p for p in m.parameters()]
def rel(a, b): a = a.float(); b = b.float(); return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
for A, B, S, su in ((2, 1, 300, None), (3, 2, 517, [517, 400]), (4, 1, 1000, [940])):
    m, q, c1, c, cos, sin, valid, ap = make(A, B, S, su)
    dy = torch.randn_like(q) * valid[..., None]
    # fp32 reference on fp32 copies of the same bf16 values
    import copy
    m32 = copy.deepcopy(m).float()
    q32 = q.float().requires_grad_(True); c132 = c1.float().requires_grad_(True)
    r = ref_fp32(m32, q32, c132.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, S, -1), cos, sin, valid, A)
    gr = torch.autograd.grad(r, [q32, c132] + params(m32), dy.float())
    # engine
    qe = q.clone().requires_grad_(True); ce = c1.clone().requires_grad_(True)
    e = m(qe, ce.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, S, -1), ap)
    ge = torch.autograd.grad(e, [qe, ce] + params(m), dy)
    # ours
    qo = q.clone().requires_grad_(True); co = c1.clone().requires_grad_(True)
    o = ours_diff(m, qo, co, cos, sin, valid, B)
    go = torch.autograd.grad(o, [qo, co] + params(m), dy)
    names = ["q", "c"] + [n.replace("blocks.", "b").replace(".weight", "").replace("adaln_modulation.1", "mod") for n, _ in m.named_parameters()]
    print("A=%d B=%d S=%d seqused=%s | out: engine %.1e ours %.1e" % (A, B, S, su, rel(e[valid], r[valid]), rel(o[valid], r[valid])), flush=True)
    worst_e = worst_o = 0
    line = []
    for nm, a, b_, rr in zip(names, go, ge, gr):
        if nm == "q": a, b_, rr = a[valid], b_[valid], rr[valid]
        eo, ee = rel(a, rr), rel(b_, rr); worst_o = max(worst_o, eo); worst_e = max(worst_e, ee)
        line.append("%s %.1e/%.1e" % (nm, eo, ee))
    print("   grads ours/engine vs fp32: " + " ".join(line), flush=True)
    print("   worst: ours %.2e  engine %.2e" % (worst_o, worst_e), flush=True)
