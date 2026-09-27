"""Why does bench_vs_release's reference compare exactly equal?  Print what each tensor actually is."""
import copy, os, sys, torch
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "oc"))
from miniworld_engine.modules import TriangleAttention

L, C, H = 256, 128, 4
torch.manual_seed(1)
m = TriangleAttention(d_pair=C, n_head=H, starting=True, implementation="pytorch", p_drop=0.0).cuda()
for t in list(m.parameters()) + list(m.buffers()):
    t.data = t.data.to(torch.bfloat16) if t.ndim >= 2 else t.data.float()
x0 = torch.randn(1, L, L, C, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, L, device="cuda", dtype=torch.bool); mask[:, ::7] = False
with torch.no_grad():
    y = copy.deepcopy(m).float()(x0.float(), mask)
print("x0     ", tuple(x0.shape), "norm %.4f" % x0.float().norm().item())
print("fwd out", tuple(y.shape), "norm %.4f" % y.norm().item())
print("fwd out == x0 ?", bool(torch.equal(y, x0.float())))
print("||y - x0|| / ||x0|| = %.4e" % ((y - x0.float()).norm() / x0.float().norm()).item())
print("y is x0 ?", y.data_ptr() == x0.data_ptr())
print("module returns z+u ?  (u = y - x0) norm %.4f" % (y - x0.float()).norm().item())
print("params:", {n: (tuple(p.shape), str(p.dtype), round(p.float().abs().mean().item(), 6)) for n, p in list(m.named_parameters())[:6]})
