"""Wide baseline: original Triton module, complete training timings and profiles."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'bwd/baseline_comparison_20260926'))
from original_triton import TriangleAttention as OriginalTriton, METADATA, core as original_core
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.triton import main as current_core

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--length', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
C, L = args.width, args.length
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
report = dict(width=C, length=L, heads=4, head_dim=C//4, dtype='bfloat16',
              baseline='Original Triton engine', original=METADATA,
              device=torch.cuda.get_device_name(), torch=torch.__version__, measurements={})


def save(): args.output.write_text(json.dumps(report, indent=2) + '\n')


def capture(fn, stream=None):
    stream = stream or torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): out = fn()
    torch.cuda.synchronize()
    del out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): out = fn()
    graph.replay(); torch.cuda.synchronize()
    return graph, out


def measure(fn, stream=None):
    graph, out = capture(fn, stream)
    values = out if isinstance(out, (tuple, list)) else (out,)
    assert all(torch.isfinite(t).all() for t in values)
    for _ in range(10): graph.replay()
    times = []
    for _ in range(12):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(10): graph.replay()
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end) / 10)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3): graph.replay()
        torch.cuda.synchronize()
    kernels = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            cell = kernels.setdefault(event.name, dict(us=0, calls=0))
            cell['us'] += event.device_time_total / 3
            cell['calls'] += 1
    assert any('_attn_' in name for name in kernels)
    return dict(ms=statistics.median(times), rounds_ms=times,
                kernels=sorted([dict(name=name, **value) for name, value in kernels.items()], key=lambda x:-x['us']))


for ending in (False, True):
    torch.manual_seed(92691)
    model = OriginalTriton(C, n_head=4, d_hidden=C, starting=not ending,
                          implementation='triton', p_drop=0).cuda().bfloat16().train()
    current = TriangleAttention(C, n_head=4, d_hidden=C, starting=not ending,
                                implementation='triton', p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if weight.ndim >= 2: weight.normal_(std=C**-.5)
            elif name.endswith('weight'): weight.normal_(mean=1, std=.05)
            else: weight.normal_(std=.05)
    current.load_state_dict(model.state_dict())
    x = torch.randn(1,L,L,C,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy = torch.randn_like(x)
    mask = torch.ones(1,L,device='cuda',dtype=torch.bool); mask[:,::7] = False
    params = (x,*model.parameters())
    y = model(x,mask)
    old = torch.autograd.grad(y,params,dy,retain_graph=True)
    cy = current(x,mask)
    new = torch.autograd.grad(cy,(x,*current.parameters()),dy)
    output_error=float((y.detach().float()-cy.detach().float()).norm()/y.detach().float().norm().clamp_min(1e-8))
    assert output_error<.015,output_error
    errors = {name: dict(bitwise=torch.equal(a,b),
                        relative_l2=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)))
              for name,a,b in zip(('input',*dict(model.named_parameters())),old,new)}
    print('CURRENT_VS_ORIGINAL',C,L,ending,errors,flush=True)
    # The original/current namespaces independently select Triton autotune tiles;
    # tile-dependent BF16 rounding and atomic LN affine sums need not be bitwise.
    # Use the established whole-module comparison tolerance and record every error.
    for name,a,b in zip(('input',*dict(model.named_parameters())),old,new):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert errors[name]['relative_l2'] < .015, (name, errors[name])
    configs={arm:{name:str(getattr(getattr(module,name),'best_config',None))
                  for name in ('_attn_fwd','_attn_bwd_preprocess','_attn_bwd_dq','_attn_bwd_dkdv')}
             for arm,module in (('original',original_core),('current',current_core))}
    report.setdefault('current_matches_original', {})[str(ending)] = dict(output_relative_l2=output_error,gradients=errors,configs=configs)
    print('AUTOTUNE_CONFIGS',configs,flush=True)
    state = model.state_dict()
    input_value = x.detach()
    del old,new,cy,current,y,params,model,x
    gc.collect(); torch.cuda.empty_cache()
    for regime in ('forward','backward','forward_backward'):
        # Fresh leaves avoid autograd stream metadata left by the independent
        # eager numerical comparison. All timing forwards/backwards use one stream.
        model = OriginalTriton(C, n_head=4, d_hidden=C, starting=not ending,
                              implementation='triton', p_drop=0).cuda().bfloat16().train()
        model.load_state_dict(state)
        x = input_value.clone().requires_grad_()
        params = (x,*model.parameters())
        stream = None
        if regime == 'forward':
            def step(): return model(x,mask)
        elif regime == 'backward':
            # PyTorch backward follows the streams recorded by its forward.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): y = model(x,mask)
            torch.cuda.current_stream().wait_stream(stream)
            def step(): return torch.autograd.grad(y,params,dy,retain_graph=True)
        else:
            def step(): return torch.autograd.grad(model(x,mask),params,dy)
        key = f'{regime}_e{int(ending)}'
        print('MEASURE', C,L,key,flush=True)
        report['measurements'][key] = measure(step,stream)
        save()
        print('RESULT',C,L,key,report['measurements'][key]['ms'],flush=True)
        if regime == 'backward': del y
        del model,x,params,step
        gc.collect(); torch.cuda.empty_cache()
    del state,input_value,dy,mask
    gc.collect(); torch.cuda.empty_cache()
report['complete'] = True
save()
