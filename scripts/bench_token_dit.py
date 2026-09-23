"""Token DiT micro-benchmark: v1 ``DiffusionTransformer`` (augmented pair-bias attention)
vs v2.0.0 ``BiasOnlyTokenDiT``, identical dims and inputs, fwd and fwd+bwd, two precision
policies (fp32 params + bf16 autocast = today's DiT; bf16 params = v2.0.0 step 2), eager
and torch.compile(dynamic=False). Random inputs -- shapes decide the cost, not values.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from team_gm.modules import DiffusionTransformer
from team_gm.modules.exceptions import ImplementationType

from miniworld.modules.bias_only_token_dit import BiasOnlyTokenDiT


def timed(fn, warm, rep):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def build(kind, impl, ckpt):
    common = dict(d_single=768, d_cond=384, d_pair=128, n_head=16, n_block=24,
                  n_checkpoint_segments=ckpt, implementation=impl)
    if kind == "augmented":
        return DiffusionTransformer(DiffusionTransformer.Config(use_qk_norm=True, **common))
    return BiasOnlyTokenDiT(BiasOnlyTokenDiT.Config(**common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--A", type=int, default=48)
    ap.add_argument("--ckpt", type=int, default=24, help="n_checkpoint_segments (production: 24)")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rep", type=int, default=7)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    torch.set_float32_matmul_precision("medium")  # as the trainer does (TF32 for fp32 GEMMs)
    impl = ImplementationType("miniworld_engine")
    torch.manual_seed(0)
    A, B, L = args.A, 1, args.L
    single0 = torch.randn(A, B, L, 768, device=dev)
    cond0 = torch.randn(A, B, L, 384, device=dev)
    pair0 = torch.randn(B, L, L, 128, device=dev)
    mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    results = {"L": L, "A": A, "ckpt": args.ckpt, "rows": []}
    print(f"L={L} A={A} n_block=24 ckpt={args.ckpt}", flush=True)
    print(f"{'kind':10s} {'params':13s} {'mode':9s} {'fwd (s)':>9s} {'fwd+bwd (s)':>12s} {'peak GiB':>9s}")
    for kind in ("augmented", "bias_only"):
        # fp32          : PRODUCTION v1 (Fabric 32-true, no autocast; fp32 params, TF32 GEMMs)
        # fp32+autocast : fp32 params under bf16 autocast (what the earlier profiles used)
        # bf16          : full bf16 params/activations, norms fp32 (v2.0.0 policy)
        for pdtype in ("fp32", "fp32+autocast", "bf16"):
            torch.manual_seed(0)
            mod = build(kind, impl, args.ckpt).to(dev)
            if pdtype == "bf16":
                mod = mod.to(torch.bfloat16)   # engine norms (incl. patched AdaLN) stay fp32
                single, cond, pair = (t.to(torch.bfloat16) for t in (single0, cond0, pair0))
                ctx = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
            elif pdtype == "fp32+autocast":
                single, cond, pair = single0, cond0, pair0
                ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
            else:
                single, cond, pair = single0, cond0, pair0
                ctx = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
            single_g = single.clone().requires_grad_(True)
            for mode in ("eager", "compiled"):
                fn = mod
                if mode == "compiled":
                    torch._dynamo.reset()
                    fn = torch.compile(mod, dynamic=False)

                def fwd():
                    with torch.no_grad(), ctx():
                        fn(single, cond, pair, mask=mask)

                def fwd_bwd():
                    mod.zero_grad(set_to_none=True)
                    with ctx():
                        out = fn(single_g, cond, pair, mask=mask)
                    out.float().pow(2).mean().backward()
                try:
                    torch.cuda.reset_peak_memory_stats()
                    f = timed(fwd, args.warm, args.rep)
                    fb = timed(fwd_bwd, args.warm, args.rep)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    row = dict(kind=kind, params=pdtype, mode=mode, fwd=f, fwd_bwd=fb, peak_gib=peak)
                    print(f"{kind:10s} {pdtype:13s} {mode:9s} {f:9.3f} {fb:12.3f} {peak:9.1f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    row = dict(kind=kind, params=pdtype, mode=mode, error=f"{type(exc).__name__}: {exc}"[:300])
                    print(f"{kind:10s} {pdtype:13s} {mode:9s} FAILED {row['error']}", flush=True)
                results["rows"].append(row)
            del mod, fn
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
