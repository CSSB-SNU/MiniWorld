"""TriangleAttention block: the Anthropic release as shipped vs this tree, same shapes, same call, same process.

    python bench_vs_release.py --oc <dir containing opt_core> --tag release --length 768

`--oc` decides which opt_core is imported, so the two trees are compared by running them, not by reading diffs.  The call is
the in-place residual block (`tri_attn_block(..., residual=True)`) -- the form both trees implement -- plus, when the tree
supports it, the out-of-place form the engine actually wants (`out=`).  Reports op time (CUDA graph), the per-kernel
breakdown, and rel_rms against an fp32 reference of the same module.
"""
import argparse, copy, json, os, statistics, sys, torch

p = argparse.ArgumentParser()
p.add_argument("--oc", required=True, help="directory that contains the opt_core package to import")
p.add_argument("--tag", required=True)
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--output", default="")
a = p.parse_args()

R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(a.oc))                      # FIRST on the path: this decides which tree is measured
import opt_core                                                 # noqa: E402
assert os.path.abspath(opt_core.__file__).startswith(os.path.abspath(a.oc) + os.sep), (a.oc, opt_core.__file__)
from opt_core.attn import pair_fused as pf                      # noqa: E402
from miniworld_engine.modules import TriangleAttention          # noqa: E402

torch.manual_seed(20260921)
m = TriangleAttention(d_pair=a.width, n_head=a.n_head, starting=True, implementation="pytorch", p_drop=0.0).to("cuda").eval()
for n_, t in m.named_parameters():                                       # the shipped module zero-inits to_out, which would make
    if t.ndim >= 2:                                                      # EVERY comparison pass on an all-zero update: draw real weights
        t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / t.shape[-1] ** 0.5)
    else:
        t.data = t.data.float()
        t.data.normal_(1, .05) if n_.endswith("weight") else t.data.normal_(0, .05)
x0 = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool); mask[:, ::7] = False
with torch.no_grad():
    fwd = copy.deepcopy(m).float()(x0.float(), mask)                     # this module OWNS its residual: forward(z) = z + u
    upd, ref_sum = fwd - x0.float(), fwd
    assert float(upd.norm()) > 0, "the reference update is identically zero -- the weights make this comparison vacuous"
W = pf.pack_triattn_weights(w_o=m.to_out.weight, w_q=m.to_query.weight, w_k=m.to_key.weight, w_v=m.to_value.weight,
                            w_g=m.to_gate.weight, w_b=m.to_bias.weight, ln_w=m.ln_pair.weight, ln_b=m.ln_pair.bias,
                            n_heads=m.n_head, head_dim=m.to_value.weight.shape[0] // m.n_head, eps=m.ln_pair.eps)
m5 = mask[:, None, None, None, :].expand(-1, x0.shape[1], -1, -1, -1)
KW = dict(ending=False, impl="fpf", core="tier:triattn_native", ln="fused")
x = x0.clone()
buf = torch.empty_like(x0)


def inplace():
    x.copy_(x0)                                                          # the residual block consumes z in place
    return pf.tri_attn_block(x, W, m5, residual=True, **KW)


def outofplace():
    return pf.tri_attn_block(x0, W, m5, residual=True, out=buf, **KW)


def cupti(fn, iters):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            per[e.name[:44]] = per.get(e.name[:44], 0.0) + t
    return {k: v / iters for k, v in per.items() if "copy" not in k.lower()}


def graph_us(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    o = []
    for _ in range(3):
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(100):
            g.replay()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1000 / 100)
    return statistics.median(o)


res = {"tag": a.tag, "oc": os.path.abspath(opt_core.__file__), "length": a.length, "cells_sha": pf.cells_sha256()[:12]}
with torch.no_grad():
    for name, fn in (("inplace", inplace), ("outofplace", outofplace)):
        try:
            y = fn()
        except Exception as e:                                            # a form this tree does not implement
            res[name] = {"unsupported": "%s: %s" % (type(e).__name__, str(e)[:120])}
            print("%-10s %-11s UNSUPPORTED %s" % (a.tag, name, str(e)[:100]), flush=True); continue
        torch.cuda.synchronize()
        cands = {"z+u@buffer": (x if name == "inplace" else buf).float()}
        if isinstance(y, torch.Tensor):
            cands["returned"] = y.float()
        allc = [(float((g_.reshape(r.shape) - r).square().mean().sqrt() / r.square().mean().sqrt()), "%s~%s" % (w, rn))
                for w, g_ in cands.items() if g_ is not None
                for rn, r in (("z+u", ref_sum), ("u", upd)) if g_.numel() == r.numel()]
        print("      candidates: " + ", ".join("%s %.3e" % (n, e) for e, n in allc), flush=True)
        err, against = min(allc, key=lambda t: t[0])
        per = cupti(fn, a.iters)
        r = {"op_us": graph_us(fn), "rel_rms": err,
             "prologue_us": sum(v for k, v in per.items() if "prologue" in k),
             "epilogue_us": sum(v for k, v in per.items() if "epilogue" in k),
             "core_us": sum(v for k, v in per.items() if "triattn_m1_kernel" in k or k in ("_fwd", "_flash_triattn_fwd")),
             "kernels": {k: round(v, 1) for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:7]},
             "matched": against,
             "fell_back": any("flash" in k for k in per)}
        res[name] = r
        print("%-10s %-11s op %8.1f  pro %7.1f  core %7.1f  epi %7.1f  rel_rms %.4e (%s)%s"
              % (a.tag, name, r["op_us"], r["prologue_us"], r["core_us"], r["epilogue_us"], err, against,
                 "  !! FELL BACK TO FLASH" if r["fell_back"] else ""), flush=True)
print("RESULT " + json.dumps(res), flush=True)
if a.output:
    json.dump(res, open(a.output, "w"), indent=1)
