"""Prologue config sweep: run the whole block with FPF_TRIATTPRO_CFG set, read the prologue's own CUPTI time.

    FPF_TRIATTPRO_CFG=BI,BJ,W,S python sweep_pro.py --length 768 --output x.json
"""
import argparse, json, math, os, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--iters", type=int, default=25)
p.add_argument("--core", default="tier:triattn_native")
p.add_argument("--ln", default="fused")
p.add_argument("--output", required=True)
a = p.parse_args()
torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.modules import TriangleAttention  # noqa: E402
from opt_core.attn import pair_fused as pf  # noqa: E402


def init(m):
    m = m.to("cuda").eval()
    for n_, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if n_.endswith("weight") else t.data.normal_(0, .05)
    return m


m = init(TriangleAttention(d_pair=a.width, n_head=a.n_head, starting=True, implementation="pytorch", p_drop=0.0))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
import copy
with torch.no_grad():
    ref = copy.deepcopy(m).float()(x.float(), mask)
W = pf.pack_triattn_weights(w_o=m.to_out.weight, w_q=m.to_query.weight, w_k=m.to_key.weight, w_v=m.to_value.weight,
                            w_g=m.to_gate.weight, w_b=m.to_bias.weight, ln_w=m.ln_pair.weight, ln_b=m.ln_pair.bias,
                            n_heads=m.n_head, head_dim=m.to_value.weight.shape[0] // m.n_head, eps=m.ln_pair.eps)
m5 = mask[:, None, None, None, :].expand(-1, x.shape[1], -1, -1, -1)
buf = torch.empty_like(x)
KW = dict(ending=False, impl="fpf", core=a.core, ln=a.ln)


xln = torch.empty_like(x) if a.ln == "stock" else None


def call():
    if a.ln == "stock":                      # the LN the prologue would have done in-kernel, as its own pass
        xl = torch.nn.functional.layer_norm(x[0].float(), (a.width,), m.ln_pair.weight.float(),
                                            m.ln_pair.bias.float(), m.ln_pair.eps).to(torch.bfloat16)
        return pf.tri_attn_block(x, W, m5, residual=True, out=buf, x_ln=xl, **KW)
    return pf.tri_attn_block(x, W, m5, residual=True, out=buf, **KW)


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
            per[e.name[:40]] = per.get(e.name[:40], 0.0) + t
    return {k: v / iters for k, v in per.items()}


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
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(100):
            g.replay()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1000 / 100)
    return statistics.median(o)


with torch.no_grad():
    y = call()
    torch.cuda.synchronize()
    d = y.float() - ref
    err = float(d.square().mean().sqrt() / ref.square().mean().sqrt())
    per = cupti(call, a.iters)
    pro = sum(v for k, v in per.items() if "prologue" in k)
    epi = sum(v for k, v in per.items() if "epilogue" in k)
    op = graph_us(call)
res = dict(cfg=os.environ.get("FPF_TRIATTPRO_CFG", "cell"), epi_cfg=os.environ.get("FPF_TRIATT_EPI_CFG", "cell"), core=a.core, ln=a.ln, length=a.length, op_us=op, prologue_us=pro, epilogue_us=epi,
           rel_rms=err, kernels={k: round(v, 1) for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:6]})
print("RESULT", json.dumps(res), flush=True)
json.dump(res, open(a.output, "w"), indent=1)
