"""Paired graphs from the real module harness, before/after cache routing and PyTorch."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from omegaconf import OmegaConf
    from miniworld_engine.autotune import cache
    from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bd
    from miniworld_engine.kernels.layernorm_linear.triton import te_style as te
    from check_trimul_f567_runtime import paired
    spec=importlib.util.spec_from_file_location('paired_module_runner',a.run/'harness/benchmarks/runners/bench.py')
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    config=OmegaConf.to_container(OmegaConf.load(a.run/'harness/bench.yaml'))
    config.update(mode='training',min_seq_len=384,max_seq_len=384,cudagraph='manual',dropout=.25)
    conf=bench.BenchConfig(**config)
    torch.backends.cuda.matmul.allow_tf32=conf.allow_tf32
    torch.backends.cudnn.allow_tf32=conf.allow_tf32

    class NoFabric:
        @staticmethod
        def setup_module(module):return module
        @staticmethod
        def backward(tensor,gradient):tensor.backward(gradient)

    norm_fwd,norm_bwd=bd._ln_materialize,bd._te_backward
    original_subset=cache._cached_subset
    original_capture=bench.capture_cudagraph
    original_bench=bench.bench_time
    graphs,keep,results={},{},{}
    mode=None

    def subset(tuner,configs,nargs,meta):
        if mode=='triton_before' and tuner is bd._bidir_front_kernel:
            return cache.heuristic_subset(configs,24)
        return original_subset(tuner,configs,nargs,meta)

    def old_norm_fwd(*args,**kwargs):
        kwargs['shape_key']=None
        return norm_fwd(*args,**kwargs)

    def old_norm_bwd(*args,**kwargs):
        kwargs['shape_key']=None
        return norm_bwd(*args,**kwargs)

    def capture(fn,params,is_train,**kwargs):
        graph=original_capture(fn,params,is_train,**kwargs)
        graphs[mode]=graph;keep[mode]=(fn,params)
        return graph

    cache._cached_subset=subset;bench.capture_cudagraph=capture
    for mode in ['triton_before','triton_after','pytorch']:
        bd._ln_materialize=old_norm_fwd if mode=='triton_before' else norm_fwd
        bd._te_backward=old_norm_bwd if mode=='triton_before' else norm_bwd
        for kernel in [bd._bidir_front_kernel,te._ln_mat_kernel,te._ln_bwd_kernel]:kernel.cache.clear()
        torch.compiler.reset()
        with bench.forward_stream(conf):
            row=bench.bench_module_triangle_multiplication_bidirectional(
                conf,384,'pytorch' if mode=='pytorch' else 'triton',NoFabric())
        results[mode]=row._asdict()
        print('HARNESS',mode,row.value,flush=True)
    bd._ln_materialize,bd._te_backward=norm_fwd,norm_bwd
    cache._cached_subset=original_subset
    report=dict(config=conf.model_dump(),harness=results,paired=paired(graphs),
                scope='Same runner setup, compiled manual graphs; alternating order in 12 rounds of 30 replays. '
                      'Before reproduces missing F2 cache and default output-LN shape key; after uses production routing.')
    a.output.write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report['paired']),flush=True)


if __name__=='__main__':main()
