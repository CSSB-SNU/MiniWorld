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
dy = torch.randn_like(q); dev = "cuda"
def run(name, kern, grid, args, kw, cfgs):
    res = []
    for cfg in cfgs:
        try:
            f = lambda: kern.fn[grid(cfg)](*args, **kw, **cfg["meta"], num_warps=cfg["w"], num_stages=cfg.get("st", 1))
            k = f(); ms = t(f); res.append((ms, cfg, k.n_regs, k.n_spills))
        except Exception as ex:
            pass
    res.sort(key=lambda x: x[0])
    for ms, cfg, r, sp in res[:4]: print("%-10s %-40s %.3f ms regs %3d spills %4d" % (name, str(cfg["meta"]) + " w%d st%d" % (cfg["w"], cfg.get("st", 1)), ms, r, sp), flush=True)
gr = lambda cfg: (triton.cdiv(M, cfg["meta"]["BR"]),)
Qh = torch.empty(A, H, S, D, device=dev, dtype=torch.bfloat16); Kh = torch.empty_like(Qh); Vh = torch.empty_like(Qh); G = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
r1 = torch.empty(M, device=dev); Xs = torch.empty_like(G); PQs = torch.empty_like(G); PKs = torch.empty_like(G)
run("qkvg_fwd", TS._qkvg_fwd, gr, (q.reshape(M, C), mod, cs, sn, W[0], W[1], Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, TS.FP32_EPS, TS.FP32_EPS, 0),
    dict(C=C, H=H, D=D, MODW=6 * C, SAVE=True), [dict(meta=dict(BR=br), w=w, st=st) for br in (16, 32, 64) for w in (1, 2, 4, 8) for st in (1, 2, 3)])
q2 = torch.empty(M, C, device=dev, dtype=torch.bfloat16); q1 = torch.empty_like(q2); r2 = torch.empty(M, device=dev)
Att = torch.empty_like(q2); Ys = torch.empty_like(q2); ABs = torch.empty(M, 512, device=dev, dtype=torch.bfloat16); FFs = torch.empty_like(q2)
run("oproj_ffn", TS._oproj_ffn_fwd, gr, (q.reshape(M, C), sv["O"], sv["G"], mod, W[2], W[3], W[4], q1, q2, r2, Att, Ys, ABs, FFs, M, S, B, TS.FP32_EPS, 0),
    dict(C=C, NHID=NHID, MODW=6 * C, SAVE=True), [dict(meta=dict(BR=br, NH=nh), w=w, st=st) for br in (16, 32, 64) for nh in (64, 128, 256) for w in (1, 2, 4, 8) for st in (1, 2)])
grid_t = lambda cfg: (triton.cdiv(S, cfg["meta"]["AT"]), triton.cdiv(A, cfg["meta"]["SP"]), B)
dmod = torch.zeros(S, 6 * C, device=dev); dq1 = torch.empty(M, C, device=dev)
dab = torch.empty(M, 512, device=dev, dtype=torch.bfloat16); hh = torch.empty(M, 256, device=dev, dtype=torch.bfloat16); dffn = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
tiles = [(1, 16), (2, 8), (4, 4), (1, 32), (2, 16), (4, 8), (8, 4), (16, 1), (16, 2)]
run("ffn_bwd", TS._ffn_bwd, grid_t, (dy.reshape(M, C), sv["q1"], mod, W[3], W[4], sv["AB"], sv["FF"], dq1, dab, hh, dffn, dmod, S, A, B, TS.FP32_EPS, 0),
    dict(C=C, NHID=NHID, MODW=6 * C), [dict(meta=dict(SP=sp, AT=at, NH=nh), w=w) for sp, at in tiles for nh in (64, 128) for w in (1, 2, 4)])
dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dG = torch.empty_like(dO); Dv = torch.empty(A, H, S, device=dev); datt = torch.empty_like(dO); gated = torch.empty_like(dO)
run("oproj_bwd", TS._oproj_bwd, grid_t, (dq1, sv["O"], sv["G"], mod, W[2], sv["Att"], dO, dG, Dv, datt, gated, dmod, S, A, B, 0), dict(C=C, H=H, D=D, MODW=6 * C),
    [dict(meta=dict(SP=sp, AT=at), w=w) for sp, at in tiles for w in (1, 2, 4)])
dQh = torch.zeros(A, H, S, D, device=dev, dtype=torch.bfloat16); dKh = torch.zeros_like(dQh); dVh = torch.zeros_like(dQh)
dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
run("qkvg_bwd", TS._qkvg_bwd, grid_t, (q.reshape(M, C), mod, cs, sn, W[0], W[1], sv["PQ"], sv["PK"], dQh, dKh, dVh, dG, dq1, dq, dP, dmod, S, A, B, TS.FP32_EPS, TS.FP32_EPS, 0),
    dict(C=C, H=H, D=D, MODW=6 * C), [dict(meta=dict(SP=sp, AT=at), w=w) for sp, at in tiles for w in (1, 2, 4)])
