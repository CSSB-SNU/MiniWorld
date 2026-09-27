"""Register footprint of the compiled prologue kernel for the current env (FPF_TRIATTPRO_*): runs one block, then reads the Triton cache."""
import argparse, os, sys, torch
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4); a = p.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "oc"))
from miniworld_engine.modules import TriangleAttention                      # noqa: E402
from opt_core.attn import pair_fused as pf                                  # noqa: E402
from opt_core.kernels.fpf_triatt_pro import prologue as P                   # noqa: E402

m = TriangleAttention(d_pair=a.width, n_head=a.n_head, starting=True, implementation="pytorch", p_drop=0.0).cuda()
for t in list(m.parameters()) + list(m.buffers()):
    t.data = t.data.to(torch.bfloat16) if t.ndim >= 2 else t.data.float()
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool); mask[:, ::7] = False
W = pf.pack_triattn_weights(w_o=m.to_out.weight, w_q=m.to_query.weight, w_k=m.to_key.weight, w_v=m.to_value.weight,
                            w_g=m.to_gate.weight, w_b=m.to_bias.weight, ln_w=m.ln_pair.weight, ln_b=m.ln_pair.bias,
                            n_heads=m.n_head, head_dim=m.to_value.weight.shape[0] // m.n_head, eps=m.ln_pair.eps)
m5 = mask[:, None, None, None, :].expand(-1, x.shape[1], -1, -1, -1)
pf.tri_attn_block(x, W, m5, residual=True, out=torch.empty_like(x), ending=False, impl="fpf")
torch.cuda.synchronize()

for name in ("_triatt_prologue_kernel", "_triatt_prologue_kernel_padded"):
    k = getattr(P, name, None)
    if k is None:
        continue
    caches = getattr(k, "device_caches", None)
    entries = []
    if caches:
        for _dev, e in caches.items():
            entries += list((e[0] if isinstance(e, tuple) else e).values())
    else:
        entries = list((getattr(k, "cache", {}) or {}).values())
        entries = [c for d in entries for c in (d.values() if isinstance(d, dict) else [d])]
    for ck in entries:
        md = getattr(ck, "metadata", None)
        print("%-34s regs %-5s spills %-5s smem %-7s warps %s"
              % (name, getattr(ck, "n_regs", "?"), getattr(ck, "n_spills", "?"),
                 getattr(md, "shared", "?"), getattr(md, "num_warps", "?")))
print("ADDR=%s NOASM=%s" % (os.environ.get("FPF_TRIATTPRO_ADDR", "split"), os.environ.get("FPF_TRIATTPRO_NOASM", "0")))
