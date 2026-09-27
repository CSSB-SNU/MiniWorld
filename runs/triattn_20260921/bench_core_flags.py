"""A/B the cuda_b core kernel's compile-time structural variants (Traits<flags>) directly, below the package face.

    TRIATTN_M1_FLAGS="0;8;..." python bench_core_flags.py --length 768 --flags 0,8,...

Per variant: kernel time (CUDA events over the call, which is the kernel plus its tiny staging launches), bitwise equality vs
flags 0, and max|err| / rms vs an fp64 reference of pair row 0 -- so a variant that trades numerics says so in the same table.
"""
import argparse, json, os, sys, time, torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--head-dim", type=int, default=32)
p.add_argument("--rows", type=int, default=0, help="pair rows N (default: --length)")
p.add_argument("--flags", default="0")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--mask", default="prefixvar", choices=("none", "prefixvar"))
p.add_argument("--qkv-layout", default="nhsd", choices=("nhsd", "nshd", "shnd"),
               help="which PHYSICAL tensor carries the [B,N,H,S,D] logical q/k/v the core consumes:\n"
                    "  nhsd = contiguous [B,N,H,S,D] (what the prologue writes today: 64 B runs, uncoalesced)\n"
                    "  nshd = contiguous [B,N,S,H,D] handed as a permuted view (pair-natural: the prologue could write it in 256 B runs)\n"
                    "  shnd = contiguous [B,S,H,N,D] handed transposed (the package's own `endstrided` case)")
p.add_argument("--bias-dtype", default="fp32", choices=("fp32", "bf16"), help="the pair bias as handed to the kernel; its VALUES are bf16-representable either way,\n                    so bf16 changes no arithmetic -- only the bytes the core re-reads once per pair row")
p.add_argument("--output", default="")
a = p.parse_args()

R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "oc/opt_core/kernels/triattn/triattn_native/pkg/v11"))
from triattn_pkg.cuda_b.triattn_m1 import triangle_attention_m1            # noqa: E402

B, N, H, S, D = 1, (a.rows or a.length), a.n_head, a.length, a.head_dim
g = torch.Generator("cpu").manual_seed(20260921)
mk = lambda *sh, s=1.0: (torch.randn(*sh, generator=g, dtype=torch.float32) * s).cuda()
def make_qkv():
    if a.qkv_layout == "nhsd":
        return mk(B, N, H, S, D).to(torch.bfloat16)                        # contiguous in the layout the core wants
    if a.qkv_layout == "nshd":
        return mk(B, N, S, H, D).to(torch.bfloat16).permute(0, 1, 3, 2, 4)  # [B,N,S,H,D] storage -> [B,N,H,S,D] view
    return mk(B, S, H, N, D).to(torch.bfloat16).transpose(1, 3)            # the package's endstrided case


q, k, v = (make_qkv() for _ in range(3))
bias = mk(B, 1, H, S, S, s=0.5).to(torch.bfloat16)                          # the values a trunk hands: a bf16 GEMM output
bias = bias if a.bias_dtype == "bf16" else bias.float()
mask = None
if a.mask == "prefixvar":
    npad = int(round(0.07 * S))
    L = torch.randint(max(1, S - 3 * npad), S + 1, (B, N), generator=g, dtype=torch.int64)
    L[:, 0] = S
    mask = (torch.arange(S)[None, None, :] < L[:, :, None])[:, :, None, None, :].contiguous().cuda()
scale = float(D) ** -0.5


def ref_fp64(rows=1):
    idx = torch.arange(rows, device=q.device)
    lg = torch.einsum("bnhqd,bnhkd->bnhqk", q[:, idx].double() * scale, k[:, idx].double()) + bias.double()
    if mask is not None:
        lg = lg.masked_fill(~mask[:, idx], float("-inf"))
    return torch.einsum("bnhqk,bnhkd->bnhqd", lg.softmax(-1), v[:, idx].double())


def timed(fl):
    fn = lambda: triangle_attention_m1(q, k, v, bias, mask=mask, scale=scale, flags=fl)
    assert q.shape == (B, N, H, S, D), q.shape
    for _ in range(a.warmup):
        out = fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(a.iters)]
    for s_, e_ in ev:
        s_.record(); out = fn(); e_.record()
    torch.cuda.synchronize()
    ts = sorted(s_.elapsed_time(e_) * 1e3 for s_, e_ in ev)
    return ts[len(ts) // 2], out


r64 = ref_fp64()
base = None
rows = []
for fl in [int(x) for x in a.flags.split(",")]:
    try:
        us, out = timed(fl)
    except Exception as e:                                                   # a flag this build did not instantiate
        print("flags %-10d RAISED %s: %s" % (fl, type(e).__name__, str(e)[:110]), flush=True); continue
    o0 = out[:, :1].double()
    err = (o0 - r64).abs()
    bit = base is not None and bool(torch.equal(out, base[1]))
    if base is None:
        base = (us, out.clone())
    rows.append(dict(flags=fl, us=us, rel_pct=100 * (us / base[0] - 1), bitwise=bit,
                     max_abs=err.max().item(), rms=err.pow(2).mean().sqrt().item(),
                     nonfinite=int((~torch.isfinite(out)).sum())))
    print("flags %-10d %8.1f us (%+6.1f%%)  bitwise_vs_0 %-5s  max|err| %.3e  rms %.3e  nonfinite %d"
          % (fl, us, rows[-1]["rel_pct"], bit, rows[-1]["max_abs"], rows[-1]["rms"], rows[-1]["nonfinite"]), flush=True)
    print("        qkv=%s strides q %s" % (a.qkv_layout, tuple(q.stride())), flush=True)
if a.output:
    json.dump({"shape": dict(B=B, N=N, H=H, S=S, D=D, mask=a.mask, bias=a.bias_dtype, qkv=a.qkv_layout), "rows": rows}, open(a.output, "w"), indent=1)
