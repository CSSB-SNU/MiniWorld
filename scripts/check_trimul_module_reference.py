"""All module gradients versus PyTorch with the SAME explicit dropout scale."""
import argparse
import json
from pathlib import Path
import types


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--length', type=int, default=384)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ln-partial', type=Path)
    p.add_argument('--implementation', choices=['triton','miniworld'], default='triton')
    p.add_argument('--check-output-keys', action='store_true')
    p.add_argument('--expected-output-key', choices=['actual','default'], default='actual')
    args = p.parse_args()
    import torch
    if args.ln_partial:
        from trimul_ln_partial_experiment import install_from_report
        install_from_report(args.ln_partial)
    from miniworld_engine.autotune import cache
    from miniworld_engine.autotune.shape_key import both_key, pack
    from miniworld_engine.modules import BidirectionalTriangleMultiplication as Module
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    base = Module(128)
    with torch.no_grad():
        for name, value in base.named_parameters():
            if 'ln_' not in name:
                value.normal_(std=128**-.5)
    state = base.state_dict()
    length = args.length
    torch.manual_seed(1)
    x = torch.randn(1,length,length,128,device='cuda',dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    mask = torch.rand(1,length,device='cuda') > .2
    drop = (torch.rand(1,1,length,128,device='cuda')>.25).to(x.dtype)/.75
    references = {}
    report = dict(length=length, implementation=args.implementation, comparisons={}, complete=False, output_norm_cache=[])
    original_subset, original_miss = cache._cached_subset, cache._miss
    bad_keys, misses = [], []

    def observe(tuner, configs, nargs, meta):
        subset=original_subset(tuner,configs,nargs,meta)
        if tuner.fn.__name__ in ('_ln_mat_kernel','_ln_bwd_kernel','_ln_bwd_persistent'):
            key=nargs.get('shape_key',meta.get('shape_key'))
            expected_rows=length**2 if args.expected_output_key=='actual' else 0
            if key!=pack(both_key(expected_rows),N=256):
                bad_keys.append((tuner.fn.__name__,key))
            report['output_norm_cache'].append(dict(kernel=tuner.fn.__name__,shape_key=key,
                                                   candidates=len(subset or [])))
        return subset

    def miss(op,*pos,**kw):
        if op in ('layernorm_fwd_saveact_strided_triton','layernorm_bwd_atomic_strided_triton',
                  'layernorm_bwd_split_triton'):
            misses.append((op,str(pos[:3])))
        return original_miss(op,*pos,**kw)

    cache._cached_subset,cache._miss=observe,miss
    for case in ['dropout', 'zero_scale']:
        scale = drop if case == 'dropout' else torch.zeros_like(drop)
        for impl,dtype,label in [('pytorch',torch.bfloat16,'pytorch_bf16'),
                                 ('pytorch',torch.float32,'pytorch_fp32'),
                                 (args.implementation,torch.bfloat16,'actual')]:
            model = Module(128,implementation=impl).cuda().to(dtype).train()
            model.load_state_dict(state)
            def fixed_scale(self, pair, p):
                return scale.to(pair.dtype)
            model._make_drop_row_scale = types.MethodType(fixed_scale,model)
            func = torch.compile(model, dynamic=False, fullgraph=True,
                                 options={'triton.cudagraphs':False})
            inp = x.detach().to(dtype).clone().requires_grad_(True)
            y = func(inp,mask)
            y.backward(dy.to(dtype))
            values = dict(output=y.detach(), input_grad=inp.grad,
                          **{name+'.grad':v.grad for name,v in model.named_parameters()})
            assert all(v is not None and torch.isfinite(v).all() for v in values.values())
            if case=='zero_scale':
                assert torch.equal(y,inp) and torch.equal(inp.grad,dy.to(dtype))
                assert all(torch.count_nonzero(v.grad)==0 for v in model.parameters())
            if label.startswith('pytorch'):
                references[label]={k:v.detach().float().cpu() for k,v in values.items()}
            else:
                errors={ref:{k:float((v.detach().float().cpu()-refs[k]).norm()/refs[k].norm().clamp_min(1e-8))
                             for k,v in values.items()} for ref,refs in references.items()}
                assert max(max(e.values()) for e in errors.values())<.02, errors
                report['comparisons'][case]=errors
            del y,inp,values,model,func
            torch.compiler.reset()
        references.clear()
    if args.implementation=='triton' or args.check_output_keys:
        assert report['output_norm_cache'] and not bad_keys, bad_keys
        assert not misses, misses
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
