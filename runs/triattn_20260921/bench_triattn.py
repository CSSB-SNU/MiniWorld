"""TriangleAttention through Anthropic's fused pair block (prologue + core + epilogue), per kernel.

    TRIMUL_NATIVE_BUILD_DIR unused here; needs MINIWORLD_ANTHROPIC_ROOT (env.sh sets it)
    python bench_triattn.py --length 768 --output ta-L768.json [--row block:triattn_native] [--profile-one]

Reports: per-kernel CUPTI device time grouped into prologue / core / epilogue / other, the whole-op time as a one-call CUDA
graph, and the error against the module's own fp32 reference. --profile-one brackets a single call with
cudaProfilerStart/Stop for ncu --profile-from-start off.
"""
import argparse, copy, json, math, statistics, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--row", default="block:triattn_native")
p.add_argument("--impl", default="anthropic")
p.add_argument("--starting", type=int, default=1)
p.add_argument("--iters", type=int, default=40)
p.add_argument("--output", required=True)
p.add_argument("--profile-one", action="store_true")
a = p.parse_args()

torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.modules import TriangleAttention  # noqa: E402


def init(m):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


def make(impl):
    kw = dict(d_pair=a.width, n_head=a.n_head, starting=bool(a.starting), implementation=impl, p_drop=0.0)
    if impl == "anthropic":
        kw["anthropic_row"] = a.row
    return TriangleAttention(**kw)


def classify(n):
    ln = n.lower()
    if "stage_bias" in ln or "prolog" in ln or "_qkv" in ln or "pack" in ln:
        return "prologue"
    if "_fwd" in ln or "attn" in ln or "flash" in ln:
        return "core"
    if "epilog" in ln or "gate" in ln or "out" in ln:
        return "epilogue"
    if "gemm" in ln or "nvjet" in ln or "cutlass" in ln:
        return "gemm"
    return "other"


def cupti(fn, iters):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per, names = {}, {}
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        per.setdefault(classify(e.name), []).append(t)
        d = names.setdefault(e.name[:60], [0, 0.0])
        d[0] += 1; d[1] += t
    groups = {k: dict(count=len(v) / iters, us=sum(v) / iters) for k, v in per.items()}
    kernels = sorted(({"name": k, "calls": v[0] / iters, "us": v[1] / iters} for k, v in names.items()),
                     key=lambda r: -r["us"])
    return groups, kernels


def graph_us(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    out = []
    for _ in range(3):
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(100):
            g.replay()
        en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000 / 100)
    return statistics.median(out)


ref_mod = init(make("pytorch"))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
with torch.no_grad():
    ref = copy.deepcopy(ref_mod).float()(x.float(), mask)
m = make(a.impl).to("cuda").eval()
m.load_state_dict(ref_mod.state_dict())
for _n, prm in m.named_parameters():
    prm.data = prm.data.to(torch.bfloat16 if prm.ndim >= 2 else torch.float32)

with torch.no_grad():
    y = m(x, mask)
    torch.cuda.synchronize()
    d = y.float() - ref
    err = dict(rel_rms=float(d.square().mean().sqrt() / ref.square().mean().sqrt()), max_abs=float(d.abs().max()),
               finite=bool(torch.isfinite(y).all()))
    if a.profile_one:
        for _ in range(6):
            m(x, mask)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        m(x, mask)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("PROFILED one call", flush=True)
        sys.exit(0)
    groups, kernels = cupti(lambda: m(x, mask), a.iters)
    op = graph_us(lambda: m(x, mask))
res = dict(length=a.length, width=a.width, n_head=a.n_head, row=a.row, impl=a.impl, starting=bool(a.starting),
           error=err, op_us=op, groups=groups, kernels=kernels,
           selection=str(getattr(m, "anthropic_selection", None))[:2000])
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "selection"})[:1500], flush=True)
json.dump(res, open(a.output, "w"), indent=1)
