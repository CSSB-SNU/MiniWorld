"""Verify the rewired ops.transition (LN folded into fused kernel) matches the OLD
path (F.layer_norm + split triton_transition) numerically — forward + all grads."""
import torch
import torch.nn.functional as F
from miniworld_kernels import ops
from miniworld_kernels.kernels.transition.triton.main import triton_transition

torch.manual_seed(0)
dev = "cuda"

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

for d in (128, 256):
    for M in (256*256,):
        n = 4
        x0 = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
        lw = torch.randn(d, device=dev, dtype=torch.bfloat16)
        lb = torch.randn(d, device=dev, dtype=torch.bfloat16)
        ea = torch.randn(n*d, d, device=dev, dtype=torch.bfloat16) * 0.05
        eb = torch.randn(n*d, d, device=dev, dtype=torch.bfloat16) * 0.05
        sq = torch.randn(d, n*d, device=dev, dtype=torch.bfloat16) * 0.05
        go = torch.randn(M, d, device=dev, dtype=torch.bfloat16)

        def run(use_new):
            xs = [t.clone().detach().requires_grad_(True) for t in (x0, lw, lb, ea, eb, sq)]
            x, w, b, a1, b1, s = xs
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_new:
                    out = ops.transition(x, ln_in_weight=w, ln_in_bias=b,
                                         expand_a_weight=a1, expand_b_weight=b1,
                                         squeeze_weight=s, n=n)
                else:
                    xn = F.layer_norm(x, (d,), w, b, 1e-5)
                    out = triton_transition(xn, a1, b1, s, n)
            out.backward(go)
            return out, [t.grad for t in xs]

        o_old, g_old = run(False)
        o_new, g_new = run(True)
        names = ["x", "ln_w", "ln_b", "ea", "eb", "sq"]
        gcos = {nm: cos(a, b) for nm, a, b in zip(names, g_old, g_new)}
        print(f"d={d} M={M}: out_cos={cos(o_old,o_new):.5f}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in gcos.items()))
print("VERIFY DONE")
