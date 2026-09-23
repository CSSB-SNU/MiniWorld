"""Publish measured checkpoint candidates with their exact physical workload.

Coverage remains partial. No timing is assigned to an unmeasured config.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    relative = 'kernels/layernorm/triton/persistent.py'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(run/'package/miniworld_engine'/relative) == sha(run/'baseline_package/miniworld_engine'/relative)
    import torch
    from miniworld_engine.autotune import cache
    from miniworld_engine.autotune.configs import op_of
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd as mod
    checkpoint = json.loads((run/'tune_lnout_768.progress.json').read_text())
    assert checkpoint['length'] == 768 and checkpoint['op'] == 'lnout'
    kernel = mod._ln_bwd_persistent_jit
    configs = {cache._sig(c): c for c in kernel.configs}
    measured = [(configs[cache._sig_from_dict(r['config'])], float(r['timings'][0]))
                for r in checkpoint['configs']]
    ranked = sorted(((c,t) for c,t in measured if math.isfinite(t)), key=lambda ct: ct[1])
    metadata = {}
    original = cache._cached_subset
    def capture(k, cfgs, nargs, meta):
        if k is not kernel:
            return original(k, cfgs, nargs, meta)
        op = op_of(k.configs)
        metadata.update(op=op, dtype=cache.dtype_of_args(nargs),
                        bucket=cache.bucket_of_autotuner(k, nargs, meta),
                        measurement=cache.measurement_workload(op, k, nargs, meta))
        return [ranked[0][0]]
    m, n = 768**2, 256
    torch.manual_seed(64)
    x = torch.randn(n,m,device='cuda',dtype=torch.bfloat16).t()
    dy = torch.randn_like(x); g = torch.randn(n,device='cuda')
    mean = x.float().mean(1); rs = torch.rsqrt(x.float().var(1,unbiased=False)+1e-5)
    reference = mod._ln_bwd_atomic(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
    cache._cached_subset = capture
    try:
        output = mod._ln_bwd_persistent_canonical(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m))
    finally:
        cache._cached_subset = original
    errors = [float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))
              for a,b in zip(output,reference)]
    assert metadata and max(errors) < .002 and all(torch.isfinite(a).all() for a in output)
    path = cache.store_ranked_configs(metadata['op'], cache.gpu_key(), metadata['dtype'],
        metadata['bucket'], ranked, cache.config_space_hash(kernel.configs),
        op_id=cache.op_identity(kernel), configs=kernel.configs,
        entry_configs=[c for c,_ in measured], measurement=metadata['measurement'])
    report = dict(complete=False, published=True, reason='Full search exceeded 1800 second limit',
                  measured=len(measured), grid_count=len(kernel.configs), errors=errors,
                  winner=cache.as_cfg_dict(ranked[0][0]), winner_kernel_ms=ranked[0][1],
                  metadata=metadata, kernel_sha256=sha(run/'package/miniworld_engine'/relative),
                  cache=str(path))
    (run/'tune_lnout_768.recovery.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
