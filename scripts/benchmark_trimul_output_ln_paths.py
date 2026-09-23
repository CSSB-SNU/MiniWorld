"""Compare output-LN backward schedules on the actual bidirectional m-major shape."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.layernorm_linear.triton import mmajor_bwd as mod
    from check_trimul_f567_runtime import capture, paired
    torch.manual_seed(67)
    reports = []
    for length in [128, 384, 768]:
        m, n = length**2, 256
        x = torch.randn(n, m, device='cuda', dtype=torch.bfloat16).t()
        dy = torch.randn_like(x)
        g = torch.randn(n, device='cuda')
        mean = x.float().mean(1)
        rs = torch.rsqrt(x.float().var(1, unbiased=False)+1e-5)
        graphs, values = {}, {}
        for name, fn in [('atomic',mod._ln_bwd_atomic), ('specialized',mod._ln_bwd_persistent_new),
                         ('canonical',mod._ln_bwd_persistent_canonical)]:
            graphs[name], values[name] = capture(lambda: fn(dy,x,g,mean,rs,list(x.stride()),shape_key=both_key(m)))
            graphs[name].replay()
        torch.cuda.synchronize()
        errors = {name:[float((v.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))
                        for v,b in zip(out,values['atomic'])] for name,out in values.items()}
        assert max(max(e) for e in errors.values()) < .002, errors
        row = dict(length=length, errors=errors, timings=paired(graphs))
        reports.append(row)
        print('RESULT', row, flush=True)
    args.output.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
