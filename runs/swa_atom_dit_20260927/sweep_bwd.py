import os, sys, torch, triton
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_fwd.py").read().split("with torch.no_grad():")[0])
A, S, B = 48, 4096, 1
m, q, c1, c, cos, sin, valid, ap = make(A, B, S)
blk = m.blocks[0]; C = 128; H = 4; D = 32; M = A * S; NHID = 256
seq = valid.sum(-1, dtype=torch.int32)
cs, sn = cos.reshape(S, -1).contiguous(), sin.reshape(S, -1).contiguous()
mod = TS.hoist_mod(c1, blk.adaln_modulation[1].weight)
W = (blk.attn.Wqkv.weight, blk.attn.gate_proj.weight, blk.attn.out_proj.weight, blk.ffn.w_up.weight, blk.ffn.w_down.weight)
out, sv = TS.block_fwd(q, mod, cs, sn, seq, *W, B, save=True)
dy = torch.randn_like(q)
dev = "cuda"
dmod = torch.zeros(S, 6 * C, device=dev)
dq1 = torch.empty(M, C, device=dev); dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dG = torch.empty_like(dO); Dv = torch.empty(A, H, S, device=dev)
dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.bfloat16); hh = torch.empty(M, NHID, device=dev, dtype=torch.bfloat16)
dffn = torch.empty_like(dO); yn = torch.empty_like(dO); datt = torch.empty_like(dO); gated = torch.empty_like(dO)
dQh = torch.zeros(A, H, S, D, device=dev, dtype=torch.bfloat16); dKh = torch.zeros_like(dQh); dVh = torch.zeros_like(dQh)
dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16); xn = torch.empty_like(dO)
def run(name, kern, grid, args, kw, cfgs):
    for cfg in cfgs:
        try:
            f = lambda: kern.fn[grid(cfg)](*args, **kw, **cfg["meta"], num_warps=cfg["w"], num_stages=1)
            k = f(); ms = t(f)
            print("%-10s %-28s %.3f ms regs %3d spills %4d" % (name, str(cfg["meta"]) + " w%d" % cfg["w"], ms, k.n_regs, k.n_spills), flush=True)
        except Exception as ex:
            print(name, cfg, "FAIL", str(ex)[:80])
grid_t = lambda cfg: (triton.cdiv(S, cfg["meta"]["AT"]), triton.cdiv(A, cfg["meta"]["SP"]), B)
tiles = [(1, 32), (2, 16), (1, 64), (2, 32), (4, 16)]
cf = [dict(meta=dict(SP=sp, AT=at, NH=nh), w=w) for sp, at in tiles for nh in (64, 128) for w in (4, 8, 16)]
run("ffn_bwd", TS._ffn_bwd, grid_t, (dy.reshape(M, C), sv["q1"], mod, W[3], W[4], dq1, dab, hh, dffn, yn, dmod, S, A, B, TS.FP32_EPS, 0), dict(C=C, NHID=NHID, MODW=6 * C), cf)
co = [dict(meta=dict(SP=sp, AT=at), w=w) for sp, at in tiles for w in (4, 8, 16)]
run("oproj_bwd", TS._oproj_bwd, grid_t, (dq1, sv["O"], sv["G"], mod, W[2], dO, dG, Dv, datt, gated, dmod, S, A, B, 0), dict(C=C, H=H, D=D, MODW=6 * C), co)
run("qkvg_bwd", TS._qkvg_bwd, grid_t, (q.reshape(M, C), mod, cs, sn, W[0], W[1], dQh, dKh, dVh, dG, dq1, dq, dP, xn, dmod, S, A, B, TS.FP32_EPS, TS.FP32_EPS, 0),
    dict(C=C, H=H, D=D, MODW=6 * C), co)
gr = lambda cfg: (triton.cdiv(M, cfg["meta"]["BR"]),)
q2 = torch.empty(M, C, device=dev, dtype=torch.bfloat16); q1 = torch.empty_like(q2); r2 = torch.empty(M, device=dev)
run("oproj_ffn", TS._oproj_ffn_fwd, gr, (q.reshape(M, C), sv["O"], sv["G"], mod, W[2], W[3], W[4], q1, q2, r2, M, S, B, TS.FP32_EPS, 0),
    dict(C=C, NHID=NHID, MODW=6 * C, SAVE=True), [dict(meta=dict(BR=br, NH=nh), w=w) for br in (32, 64, 128) for nh in (64, 128) for w in (4, 8, 16)])
