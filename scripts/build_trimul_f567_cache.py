"""Build F567 through the engine autotuner, checking each measured configuration.

Use the staged package through PYTHONPATH. Publication uses capture.flush(), the
same cache schema, identities, grid coverage and reader as other Triton kernels.
"""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--length',type=int,required=True)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--hidden',type=int)
    p.add_argument('--compile-jobs',type=int,default=8)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ.update(MINIWORLD_DRIVER_LENGTH=str(a.length), MINIWORLD_DRIVER_WIDTH=str(a.width),
                      MINIWORLD_DRIVER_HEADS=str(a.hidden or a.width),
                      MINIWORLD_SMEM_LOG=str(a.output.with_suffix('.smem')))
    import torch
    from miniworld_engine import settings
    from miniworld_engine.autotune import capture, cache
    settings.configure(run_autotune=True,capture=True,compile_jobs=a.compile_jobs,
                       bench_clear_mb=16,bench_rep_ms=8)
    capture.install()
    from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel as kernel
    from miniworld_engine.kernels.drivers.trimul_output import output_f567_train
    torch.backends.cuda.matmul.allow_tf32=False
    original=kernel._bench
    refs=None
    checked=[]

    def bench(*args,config,**meta):
        nonlocal refs
        timings=original(*args,config=config,**meta)
        rec={'config':cache.as_cfg_dict(config),'timings':timings}
        if math.isfinite(timings[0]):
            v=kernel.nargs
            if refs is None:
                pr=(v['XN'].float()@v['WP'].float().t()).to(torch.bfloat16).float()
                ga=torch.sigmoid((v['X'].float()@v['WG'].float()).to(torch.bfloat16).float())
                rows=torch.arange(v['M'],device=pr.device)%v['L']
                yy=v['RES'].float()+pr*ga*v['DS'].float()[rows]
                refs=(yy,pr,ga)
            errors=[]
            for key,ref in zip(('Y','PROJ','GATE'),refs):
                value=v[key].float()
                relative=((value-ref).norm()/ref.norm().clamp_min(1e-8)).item()
                assert torch.isfinite(value).all() and relative<.004,(config,key,relative)
                errors.append(relative)
            rec['relative_l2']=errors
        checked.append(rec)
        if len(checked)%100==0:
            print('CHECKED',len(checked),flush=True)
        return timings

    kernel._bench=bench
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        output_f567_train()
        assert checked and any('relative_l2' in x for x in checked)
        assert not capture.record_errors(),capture.record_errors()
        capture.dump_shard(str(a.output.with_suffix('.shard.json')),unit_complete=True)
        written=capture.flush()
        result={'complete':True,'length':a.length,'width':a.width,'hidden':a.hidden or a.width,
                'grid_count':len(kernel.configs),'measured_count':len(checked),
                'winner':cache.as_cfg_dict(kernel.best_config),'configs':checked,'written':written}
        a.output.write_text(json.dumps(result,indent=2,default=str))
        print('COMPLETE',a.output,flush=True)
    finally:
        capture.shutdown_precompile()


if __name__=='__main__':
    main()
