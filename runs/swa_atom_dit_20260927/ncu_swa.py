import os, sys, torch
sys.path.insert(0, "/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927")
exec(open("/home/psk6950/MiniWorld/runs/swa_atom_dit_20260927/test_swa_bwd.py").read().split("for A, B, S, su in")[0])
A, S = 48, 4096
m, q, c1, c, cos, sin, valid, ap = make(A, 1, S)
P = params(m); dy = torch.randn_like(q)
qq = q.clone().requires_grad_(True); cc = c1.clone().requires_grad_(True)
f = lambda: torch.autograd.grad(ours_diff(m, qq, cc, cos, sin, valid, 1), [qq, cc] + P, dy)
for _ in range(3): f()          # autotune + warm
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); f(); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
