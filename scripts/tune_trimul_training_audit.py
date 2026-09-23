"""Tune the exact masked BF16 training operands through the engine CSV/cache machinery."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', choices=['front', 'lnout', 'lnfwd'], required=True)
    parser.add_argument('--length', type=int, default=384)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import triton
    from miniworld_engine import settings
    from miniworld_engine.autotune import capture, cache
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bd
    from miniworld_engine.kernels.layernorm_linear.triton import te_style as te

    torch.manual_seed(64)
    length = args.length
    m, d, h = length**2, 128, 256
    options = dict(device='cuda', dtype=torch.bfloat16)
    if args.op == 'front':
        x = torch.randn(1, length, length, d, **options)
        weights = [torch.randn(d, h, **options) / d**0.5 for _ in range(4)]
        mask = torch.rand(m, device='cuda') > .2
        call = lambda: bd.bidir_front_triton(x, *weights, pair_mask=mask)
        kernel = bd._bidir_front_kernel
    else:
        x = torch.randn(h, m, **options).t()
        dy = torch.randn(h, m, **options).t()
        gamma = torch.randn(h, device='cuda')
        mean = x.float().mean(1)
        rs = torch.rsqrt(x.float().var(1, unbiased=False) + 1e-5)
        if args.op == 'lnout':
            call = lambda: te._ln_bwd(dy, x, gamma, mean, rs, list(x.stride()), both_key(m))
            kernel = te._ln_bwd_kernel
            if m >= 300_000:
                from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd
                kernel = mmajor_bwd._ln_bwd_persistent_jit
        else:
            beta = torch.randn_like(gamma)
            call = lambda: te._ln_materialize(x, gamma, beta, 1e-5, both_key(m))
            kernel = te._ln_mat_kernel
    before = call()
    before = tuple(v.clone() for v in before)
    before_ms = triton.testing.do_bench_cudagraph(call, rep=80)
    settings.configure(run_autotune=True, capture=True, compile_jobs=8,
                       bench_clear_mb=16, bench_rep_ms=8)
    capture.install()
    capture.set_incremental(False)
    kernel.cache.clear()
    original = kernel._bench
    records = []

    def measured(*pos, config, **kw):
        kernel.__dict__['do_bench'] = lambda fn, quantiles, **_: triton.testing.do_bench_cudagraph(fn, rep=8, quantiles=quantiles)
        timing = original(*pos, config=config, **kw)
        records.append(dict(config=cache.as_cfg_dict(config), timings=timing))
        if len(records) % 64 == 0:
            print('CHECKED', len(records), flush=True)
            args.output.with_suffix('.progress.json').write_text(json.dumps(
                dict(complete=False, op=args.op, length=length, configs=records), indent=2))
        return timing

    kernel._bench = measured
    try:
        after = call()
        errors = [float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))
                  for a, b in zip(after, before)]
        assert max(errors) < .002, errors
        assert all(torch.isfinite(v).all() for v in after)
        after_ms = triton.testing.do_bench_cudagraph(call, rep=80)
        assert records and not capture.record_errors(), capture.record_errors()
        capture.dump_shard(str(args.output.with_suffix('.shard.json')), unit_complete=True)
        written = capture.flush()
        report = dict(op=args.op, complete=True, length=length, grid_count=len(kernel.configs),
                      measured=len(records), finite=sum(math.isfinite(r['timings'][0]) for r in records),
                      before_ms=before_ms, after_ms=after_ms, errors=errors,
                      winner=cache.as_cfg_dict(kernel.best_config), configs=records, written=written)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print('RESULT', json.dumps({k:v for k,v in report.items() if k!='configs'}, default=str), flush=True)
    finally:
        capture.shutdown_precompile()


if __name__ == '__main__':
    main()
