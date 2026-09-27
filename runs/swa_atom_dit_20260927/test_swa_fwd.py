"""Fused forward vs the engine SWAAtomTransformer (bf16) and an fp32 dense reference; timing."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
import triton_swa as TS
from team_gm.modules import SWAAtomTransformer
from miniworld_engine.modules.swa_atom_attention.module import build_attention_params
dev = "cuda"; d = 128
def make(A, B, S, seqused=None, seed=0):
    torch.manual_seed(seed)
    m = SWAAtomTransformer(SWAAtomTransformer.Config(d_atom=d, d_cond=d, n_block=3, n_head=4, swa_window_size=128, expansion_ratio=2)).to(dev).to(torch.bfloat16)
    for blk in m.blocks: torch.nn.init.normal_(blk.adaln_modulation[1].weight, std=0.05)
    ref_pos = torch.randn(B, S, 3, device=dev) * 5; uid = (torch.arange(S, device=dev) // 8).view(1, S).expand(B, S)
    cos, sin = m.build_rope(ref_pos, uid)
    valid = torch.ones(A * B, S, dtype=torch.bool, device=dev)
    if seqused is not None:
        for n in range(A * B): valid[n, seqused[n % len(seqused)]:] = False
    ap = build_attention_params(cos, sin, valid, A)
    q = torch.randn(A * B, S, d, device=dev, dtype=torch.bfloat16)
    c1 = torch.randn(B, S, d, device=dev, dtype=torch.bfloat16)
    c = c1.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, S, d)
    return m, q, c1, c, cos, sin, valid, ap
def ref_fp32(m, q, c, cos, sin, valid, A):
    """Dense fp32 reference of the whole stack (same weights)."""
    x = q.float(); cf = F.silu(c.float()); N, S, _ = x.shape; H, D = 4, 32
    cosr, sinr = cos.repeat(A, 1, 1), sin.repeat(A, 1, 1)
    i = torch.arange(S, device=dev); band = (i[:, None] - i[None, :]).abs() <= 64
    allowed = band[None] & valid[:, None, :] & valid[:, :, None]
    def rope(t):
        c2 = torch.cat([cosr, cosr], -1)[:, :, None]; s2 = torch.cat([sinr, sinr], -1)[:, :, None]
        t1, t2 = t.chunk(2, -1); return t * c2 + torch.cat([-t2, t1], -1) * s2
    for blk in m.blocks:
        sa, sc_, ga, sf, scf, gf = (cf @ blk.adaln_modulation[1].weight.float().t()).chunk(6, -1)
        xa = F.rms_norm(x, (d,), eps=TS.FP32_EPS) * (1 + sc_) + sa
        qkv = (xa @ blk.attn.Wqkv.weight.float().t()).view(N, S, 3, H, D)
        qh, kh, vh = qkv.unbind(2)
        qh = rope(F.rms_norm(qh, (D,), eps=TS.FP32_EPS)); kh = rope(F.rms_norm(kh, (D,), eps=TS.FP32_EPS))
        sc = torch.einsum("nihd,njhd->nhij", qh, kh) * D ** -0.5
        sc = sc.masked_fill(~allowed[:, None], -float("inf"))
        p = torch.softmax(sc, -1).nan_to_num(0.0)
        o = torch.einsum("nhij,njhd->nihd", p, vh).reshape(N, S, d) * valid[..., None]
        a = (torch.sigmoid(xa @ blk.attn.gate_proj.weight.float().t()) * o) @ blk.attn.out_proj.weight.float().t()
        x = x + ga * a
        y = F.rms_norm(x, (d,), eps=TS.FP32_EPS) * (1 + scf) + sf
        wa, wb = blk.ffn.w_up.weight.float().chunk(2, 0)
        x = x + gf * ((F.silu(y @ wa.t()) * (y @ wb.t())) @ blk.ffn.w_down.weight.float().t())
    return x
def ours(m, q, c1, cos, sin, valid, B):
    seq = valid.sum(-1, dtype=torch.int32); S = q.shape[1]
    cs, sn = cos.reshape(B * S, -1).contiguous(), sin.reshape(B * S, -1).contiguous()
    x = q
    for blk in m.blocks:
        mod = TS.hoist_mod(c1, blk.adaln_modulation[1].weight)
        x = TS.block_fwd(x, mod, cs, sn, seq, blk.attn.Wqkv.weight, blk.attn.gate_proj.weight, blk.attn.out_proj.weight, blk.ffn.w_up.weight, blk.ffn.w_down.weight, B)
    return x
def rel(a, b, valid): a = a.float()[valid]; b = b.float()[valid]; return ((a - b).norm() / b.norm()).item()
def t(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
with torch.no_grad():
    for A, B, S, su in ((2, 1, 300, None), (3, 2, 517, [517, 400]), (5, 1, 1000, [940])):
        m, q, c1, c, cos, sin, valid, ap = make(A, B, S, su)
        r = ref_fp32(m, q, c, cos, sin, valid, A)
        e = m(q, c, ap); o = ours(m, q, c1, cos, sin, valid, B)
        print("A=%d B=%d S=%d seqused=%s | engine vs fp32 %.2e | ours vs fp32 %.2e | ours vs engine %.2e" % (A, B, S, su, rel(e, r, valid), rel(o, r, valid), rel(o, e, valid)), flush=True)
    for A, S in ((5, 4096), (48, 4096)):
        m, q, c1, c, cos, sin, valid, ap = make(A, 1, S)
        te = t(lambda: m(q, c, ap)); to = t(lambda: ours(m, q, c1, cos, sin, valid, 1))
        print("A=%d S=%d fwd: engine %.3f ms | ours %.3f ms (%.2fx) | kernels %s" % (A, S, te, to, te / to,
              {k: str(getattr(getattr(TS, k), "best_config", None)) for k in ("_qkvg_fwd", "_attn_fwd", "_oproj_ffn_fwd")}), flush=True)
