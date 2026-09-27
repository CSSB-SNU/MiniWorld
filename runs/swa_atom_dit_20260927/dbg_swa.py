import os, sys, torch
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_fwd.py").read().split("with torch.no_grad():")[0])
A, B, S = 2, 1, 300
m, q, c1, c, cos, sin, valid, ap = make(A, B, S)
blk = m.blocks[0]
seq = valid.sum(-1, dtype=torch.int32)
cs, sn = cos.reshape(B * S, -1).contiguous(), sin.reshape(B * S, -1).contiguous()
mod = TS.hoist_mod(c1, blk.adaln_modulation[1].weight)
W = (blk.attn.Wqkv.weight, blk.attn.gate_proj.weight, blk.attn.out_proj.weight, blk.ffn.w_up.weight, blk.ffn.w_down.weight)
out, sv = TS.block_fwd(q, mod, cs, sn, seq, *W, B, save=True); torch.cuda.synchronize(); print("fwd ok", flush=True)
for k, v in sv.items(): print(k, tuple(v.shape), v.dtype)
import triton, subprocess
M = A * S; C = 128; NHID = 256
dq1 = torch.empty(M, C, device="cuda"); dab = torch.empty(M, 512, device="cuda", dtype=torch.bfloat16); hh = torch.empty(M, 256, device="cuda", dtype=torch.bfloat16)
dffn = torch.empty(M, C, device="cuda", dtype=torch.bfloat16); dmod = torch.zeros(S, 768, device="cuda")
for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16), (1, 32)):
    for nh in (32, 64, 128):
        for w in (4, 8):
            print("try", sp, at, nh, w, flush=True)
            TS._ffn_bwd.fn[(triton.cdiv(S, at), triton.cdiv(A, sp), B)](torch.randn_like(q).reshape(M, C), sv["q1"], mod, W[3], W[4], sv["AB"], sv["FF"], dq1, dab, hh,
                dffn, dmod, S, A, B, TS.FP32_EPS, 0, C=C, NHID=NHID, MODW=768, SP=sp, AT=at, NH=nh, num_warps=w, num_stages=1)
            torch.cuda.synchronize()
print("all ok")
