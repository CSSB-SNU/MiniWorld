"""Does folding the block's residual into the epilogue pay, and does it compute the same thing?

Three ways to get `pair + update`, same weights and inputs, on one clock:
  add   : residual=False + a torch elementwise add   (what the engine adapter does today)
  inplace: residual=True                             (upstream's block mode; mutates the input)
  out   : residual=True with out=<fresh buffer>      (this experiment: same kernel, a different destination)

    MINIWORLD_ANTHROPIC_ROOT=<patched opt_core parent> python bench_residual.py --length 768 --output r-L768.json
"""
import argparse, copy, json, math, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--core", default="tier:triattn_native")
p.add_argument("--output", required=True)
a = p.parse_args()
torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from miniworld_engine.modules import TriangleAttention  # noqa: E402
from opt_core.attn import pair_fused as pf  # noqa: E402


def init(m):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16); t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float(); t.data.normal_(1, .05) if name.endswith("weight") else t.data.normal_(0, .05)
    return m


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


m = init(TriangleAttention(d_pair=a.width, n_head=a.n_head, starting=True, implementation="pytorch", p_drop=0.0))
x = torch.randn(1, a.length, a.length, a.width, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, a.length, device="cuda", dtype=torch.bool)
mask[:, ::7] = False
with torch.no_grad():
    ref = copy.deepcopy(m).float()(x.float(), mask)
W = pf.pack_triattn_weights(w_o=m.to_out.weight, w_q=m.to_query.weight, w_k=m.to_key.weight, w_v=m.to_value.weight,
                            w_g=m.to_gate.weight, w_b=m.to_bias.weight, ln_w=m.ln_pair.weight, ln_b=m.ln_pair.bias,
                            n_heads=m.n_head, head_dim=m.to_value.weight.shape[0] // m.n_head, eps=m.ln_pair.eps)
m5 = mask[:, None, None, None, :].expand(-1, x.shape[1], -1, -1, -1)
KW = dict(ending=False, impl="fpf", core=a.core, ln="fused")
buf = torch.empty_like(x)
res = {}
with torch.no_grad():
    def add():
        return x + pf.tri_attn_block(x, W, m5, residual=False, **KW)

    def out_():
        return pf.tri_attn_block(x, W, m5, residual=True, out=buf, **KW)

    ys = {"add": add(), "out": out_()}
    xi = x.clone()
    pf.tri_attn_block(xi, W, m5, residual=True, **KW)
    ys["inplace"] = xi
    torch.cuda.synchronize()
    for k, y in ys.items():
        d = y.float() - ref
        res[k] = dict(rel_rms=float(d.square().mean().sqrt() / ref.square().mean().sqrt()),
                      max_abs=float(d.abs().max()), finite=bool(torch.isfinite(y).all()))
    res["out_equals_add"] = bool(torch.equal(ys["out"], ys["add"]))
    res["out_equals_inplace"] = bool(torch.equal(ys["out"], ys["inplace"]))
    res["x_untouched_by_out"] = bool(torch.equal(x, x))          # x is the live input; out mode must not write it
    x0 = x.clone()
    out_()
    torch.cuda.synchronize()
    res["x_untouched_by_out"] = bool(torch.equal(x, x0))
    res["us_add"] = graph_us(add)
    res["us_out"] = graph_us(out_)
res["gain_us"] = res["us_add"] - res["us_out"]
res.update(length=a.length, width=a.width, n_head=a.n_head, core=a.core)
print("RESULT", json.dumps(res), flush=True)
json.dump(res, open(a.output, "w"), indent=1)
