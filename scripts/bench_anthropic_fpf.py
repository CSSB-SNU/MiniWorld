"""Anthropic FlashPairformer (uplifting-biomolecular-modeling opt_core) vs miniworld-engine vs cuEquivariance.

Same process, same inputs, same timing method as opt_core's cell tables (CUDA events, warm-up +
interleaved rounds, median), inference forward only (the opt_core rows are forward-only).

  --family triattn : op level, cuEquivariance calling convention. q/k/v [B,N,H,S,D] bf16, bias [B,1,H,S,S]
                     fp32, key mask [B,N,1,1,S] bool (every 3rd token masked). Engine arm = its primitive
                     kernel on the same values in its own (B,H,L,L,D) layout with the mask folded into a bf16
                     bias, exactly as the engine module does. Also the two module-level arms for context.
  --family trimul  : module level (LN_in -> projections/gates -> contraction -> LN_out -> gated out + residual),
                     weights shared across arms (engine module parameters mapped onto opt_core's ten keys).
"""
import argparse, faulthandler, json, os, statistics, sys, time, traceback
faulthandler.enable()
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--family", choices=["triattn", "trimul"], required=True)
p.add_argument("--lengths", type=int, nargs="+", default=[256, 384, 512, 768, 1024, 1536, 2048])
p.add_argument("--output", required=True)
p.add_argument("--rows", default="", help="comma list of opt_core rows to time (default: family default)")
p.add_argument("--rounds", type=int, default=7)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--native-build-dir", default="", help="trimul: also time the trimul_native payload face on cubins from this build dir (rebuilt locally)")
p.add_argument("--native-python", default="", help="trimul: the payload python/ dir holding trimul_native (for --native-build-dir)")
args = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
results = []
OUT = Path(args.output)


def save():
    OUT.write_text(json.dumps(results, indent=1))


def time_arms(arms, warmup, rounds, reps):
    """arms: {name: fn}. Interleaved ABBA rounds; returns {name: (median_ms, min_ms, samples)}."""
    for _ in range(warmup):
        for fn in arms.values():
            fn()
    torch.cuda.synchronize()
    samples = {n: [] for n in arms}
    order = list(arms)
    for r in range(rounds):
        seq = order if r % 2 == 0 else order[::-1]
        for n in seq:
            fn = arms[n]
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                fn()
            e.record(); e.synchronize()
            samples[n].append(s.elapsed_time(e) / reps)
    return {n: (statistics.median(v), min(v), v) for n, v in samples.items()}


def rel_rms(a, ref):
    a = a.float(); ref = ref.float()
    return float(((a - ref).pow(2).mean().sqrt() / (ref.pow(2).mean().sqrt() + 1e-30)))


def row_record(family, L, arm, ms=None, mn=None, samples=None, err=None, note=None, row=None):
    d = dict(family=family, length=L, arm=arm, ms=ms, min_ms=mn, samples_ms=samples, rel_rms_vs_cueq=err, note=note, served_row=row,
             gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    results.append(d); print(json.dumps({k: v for k, v in d.items() if k != "samples_ms"}), flush=True); save()


# ------------------------------------------------------------------------------------------------------------ triattn
def run_triattn(L):
    from opt_core.kernels import triattn as OT
    import cuequivariance_torch as cqt
    from miniworld_engine.kernels.triangle_attention.triton.main import triton_triangle_attention_pair_bias
    from miniworld_engine.modules import TriangleAttention
    B, N, S, H, D = 1, L, L, 4, 32
    g = torch.Generator(device=dev).manual_seed(681 + L)
    q = torch.randn(B, N, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, N, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, N, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    bias = (torch.randn(B, 1, H, S, S, device=dev, dtype=torch.float32, generator=g) * 0.5)
    resmask = torch.ones(S, device=dev, dtype=torch.bool); resmask[::3] = False
    mask = resmask.view(1, 1, 1, 1, S).expand(B, N, 1, 1, S).contiguous()
    scale = D ** -0.5

    def cueq():
        return cqt.triangle_attention(q, k, v, bias, mask, scale)

    # engine primitive: (B,H,L,L,D) layout, mask folded into a bf16 bias (production path of the module)
    qe, ke, ve = (t.permute(0, 2, 1, 3, 4).contiguous() for t in (q, k, v))
    be = bias[:, 0].to(torch.bfloat16).masked_fill(~resmask.view(1, 1, 1, S), torch.finfo(torch.bfloat16).min).contiguous()

    def engine_op():
        return triton_triangle_attention_pair_bias(qe, ke, ve, be)

    arms = {"cueq_op": cueq, "engine_op": engine_op}
    served = {}
    rows = args.rows.split(",") if args.rows else ["k2b", "flash", "cuda_sm90a", "triattn_native", "fast"]
    for r in rows:
        def mk(r=r):
            def f():
                return OT.triangle_attention(q, k, v, bias, mask, scale, word=r, stock=cqt.triangle_attention)
            return f
        try:
            sel = None
            out = mk()()
            torch.cuda.synchronize()
            arms["fpf_" + r] = mk()
            # which row actually served (the tier word 'fast' resolves per cell)
            try:
                from opt_core.kernels.triattn import cuda_sm90a as C
                stack = C.stack_key()
                sel = OT.select((9, 0), "bf16", D, H, S, "fwd", word=r, stack=stack)
                served["fpf_" + r] = sel.row
            except Exception as e:  # noqa
                served["fpf_" + r] = f"?{type(e).__name__}"
        except Exception as e:  # noqa
            row_record("triattn", L, "fpf_" + r, err=None, note=f"{type(e).__name__}: {str(e)[:300]}")
    # module-level context arms (engine TriangleAttention self-attention module, cuEq through the engine module)
    mods = {}
    for impl, name in (("miniworld", "engine_module"), ("cuequivariance", "cueq_module")):
        try:
            torch.manual_seed(681)
            m = TriangleAttention(128, n_head=4, d_hidden=128, starting=True, use_self_attention=True, implementation=impl, p_drop=0.0)
            m = m.to(dev).bfloat16().eval()
            with torch.no_grad():
                for mod in m.modules():
                    if isinstance(mod, torch.nn.Linear):
                        mod.weight.normal_(std=0.02)
            pair = torch.randn(1, L, L, 128, device=dev, dtype=torch.bfloat16, generator=g)
            mres = resmask.view(1, S)
            mods[name] = (m, pair, mres)
            def mf(m=m, pair=pair, mres=mres):
                return m(pair, mres)
            arms[name] = mf
        except Exception as e:  # noqa
            row_record("triattn", L, name, note=f"{type(e).__name__}: {str(e)[:300]}")
    with torch.no_grad():
        ref = cueq().float()
        errs = {}
        for n, fn in arms.items():
            if n.endswith("_module"):
                continue
            o = fn()
            if n == "engine_op":
                o = o.permute(0, 2, 1, 3, 4)
            errs[n] = rel_rms(o, ref)
        t = time_arms(arms, args.warmup, args.rounds, args.reps)
    for n, (med, mn, smp) in t.items():
        row_record("triattn", L, n, med, mn, smp, errs.get(n), row=served.get(n))
    # a sanity reference: fp32 torch math on the same values (rel rms of cueq itself vs it)
    with torch.no_grad():
        rows_ = [0, N // 2]                       # two sampled pair rows (the full fp32 logits tensor is O(N*S*S) = 128 GiB at 2048)
        qf, kf, vf = (x[:, rows_].float() for x in (q, k, v))
        logits = torch.einsum("bnhqd,bnhkd->bnhqk", qf * scale, kf) + bias
        logits = logits.masked_fill(~mask[:, rows_], -1e9)
        reff = torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(logits, -1), vf)
        row_record("triattn", L, "cueq_vs_fp32", err=rel_rms(ref[:, rows_], reff), note="rel rms of cueq_op against fp32 torch math (2 sampled pair rows)")
        del logits, reff


# ------------------------------------------------------------------------------------------------------------ trimul
def run_trimul(L):
    from opt_core.kernels import trimul as OM
    import cuequivariance_torch as cqt
    from miniworld_engine.modules import TriangleMultiplication
    C = 128
    g = torch.Generator(device=dev).manual_seed(681 + L)
    torch.manual_seed(681)
    eng = TriangleMultiplication(C, implementation="miniworld", p_drop=0.0).to(dev).bfloat16().eval()
    with torch.no_grad():
        for mod in eng.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.normal_(std=0.02)
        eng.ln_pair.weight.normal_(1.0, 0.05); eng.ln_pair.bias.normal_(0, 0.05)
        eng.ln_out.weight.normal_(1.0, 0.05); eng.ln_out.bias.normal_(0, 0.05)
    cue = TriangleMultiplication(C, implementation="cuequivariance", p_drop=0.0).to(dev).bfloat16().eval()
    cue.load_state_dict(eng.state_dict())
    z = torch.randn(1, L, L, C, device=dev, dtype=torch.bfloat16, generator=g)
    resmask = torch.ones(L, device=dev, dtype=torch.bool); resmask[::3] = False
    mres = resmask.view(1, L)
    mask2d = (resmask[:, None] & resmask[None, :]).to(torch.bfloat16)[None]       # [1,N,N] 0/1 for opt_core
    w = dict(ln_in_w=eng.ln_pair.weight, ln_in_b=eng.ln_pair.bias, w_ag=eng.to_left_gate.weight, w_ap=eng.to_left.weight,
             w_bg=eng.to_right_gate.weight, w_bp=eng.to_right.weight, ln_out_w=eng.ln_out.weight, ln_out_b=eng.ln_out.bias,
             w_o=eng.to_out.weight, w_og=eng.to_gate.weight)
    w = {kk: vv.detach().contiguous() for kk, vv in w.items()}

    def engine_mod():
        return eng(z, mres)

    def cueq_mod():
        return cue(z, mres)

    arms = {"cueq_module": cueq_mod, "engine_module": engine_mod}
    rows = args.rows.split(",") if args.rows else ["native", "v4", "esm_v5_fwd", "esm_shapes", "tmk3_fast", "fast"]
    caches = {}
    served = {}
    for r in rows:
        caches[r] = {}
        def mk(r=r):
            def f():
                return OM.triangle_multiplication(z, mask2d, direction="outgoing", weights=w, word=r, residual=True, cache=caches[r],
                                                  stock=cqt.triangle_multiplicative_update)
            return f
        try:
            print(f"[bench] trimul L={L}: first call of row {r!r}", flush=True)
            with torch.no_grad():
                mk()(); torch.cuda.synchronize()
            print(f"[bench] trimul L={L}: row {r!r} ok", flush=True)
            arms["fpf_" + r] = mk()
            try:
                sel = OM.select((9, 0), "bf16", C, C, L, "outgoing", word=r, stack=OM.stack_word(z) if hasattr(OM, "stack_word") else None)
                served["fpf_" + r] = sel.row
            except Exception as e:  # noqa
                served["fpf_" + r] = f"?{type(e).__name__}"
        except Exception as e:  # noqa
            row_record("trimul", L, "fpf_" + r, note=f"{type(e).__name__}: {str(e)[:300]}")
    if args.native_build_dir:
        try:
            os.environ["TRIMUL_NATIVE_BUILD_DIR"] = args.native_build_dir
            if args.native_python and args.native_python not in sys.path:
                sys.path.insert(0, args.native_python)
            from trimul_native import face as NF
            rep = NF.check(device=z.device.index, gate=False)
            ncache = {}
            def native_rebuilt():
                return NF.serve(z, mask2d, direction="outgoing", weights=w, residual=True, cache=ncache, eps=1e-5, config=None)
            print(f"[bench] trimul L={L}: payload face check ok: {str(rep)[:200]}", flush=True)
            with torch.no_grad():
                native_rebuilt(); torch.cuda.synchronize()
            arms["fpf_native_rebuilt"] = native_rebuilt
            served["fpf_native_rebuilt"] = "native(cu12.9 rebuild)"
        except Exception as e:  # noqa
            row_record("trimul", L, "fpf_native_rebuilt", note=f"{type(e).__name__}: {str(e)[:300]}")
    with torch.no_grad():
        ref = cueq_mod().float()
        errs = {n: rel_rms(fn(), ref) for n, fn in arms.items()}
        t = time_arms(arms, args.warmup, args.rounds, args.reps)
    for n, (med, mn, smp) in t.items():
        row_record("trimul", L, n, med, mn, smp, errs.get(n), row=served.get(n))
    # reference: the engine's pytorch backend in fp32 on the same weights -> how far cuEq itself is
    with torch.no_grad():
        try:
            ref32 = TriangleMultiplication(C, implementation="pytorch", p_drop=0.0).to(dev).eval()
            ref32.load_state_dict({k_: v_.float() for k_, v_ in eng.state_dict().items()})
            o32 = ref32(z.float(), mres)
            row_record("trimul", L, "cueq_vs_fp32", err=rel_rms(ref, o32), note="rel rms of cueq_module against the fp32 pytorch backend")
            del o32, ref32
        except Exception as e:  # noqa
            row_record("trimul", L, "cueq_vs_fp32", note=f"{type(e).__name__}: {str(e)[:200]}")


for L in args.lengths:
    t0 = time.time()
    try:
        (run_triattn if args.family == "triattn" else run_trimul)(L)
    except Exception as e:  # noqa
        traceback.print_exc()
        row_record(args.family, L, "ALL", note=f"{type(e).__name__}: {str(e)[:300]}")
    print(f"## L={L} done in {time.time()-t0:.1f}s", flush=True)
    torch.cuda.empty_cache()
