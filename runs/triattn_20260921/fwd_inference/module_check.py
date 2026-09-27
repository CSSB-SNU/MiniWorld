"""Compare with the real eval/no-grad module, including LN, output projection and residual."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels import _compile
from native import extension
from check import capture, paired, rel

ap = argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--length',type=int,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--rounds',type=int,default=24)
ap.add_argument('--replays',type=int,default=30)
ap.add_argument('--qualify',action='store_true')
ap.add_argument('--front',action='store_true')
ap.add_argument('--front-artifact',default='front')
ap.add_argument('--profile',action='store_true')
ap.add_argument('--profile-only',action='store_true')
ap.add_argument('--selected',action='store_true')
ap.add_argument('--out-artifact')
ap.add_argument('--default-entry',action='store_true')
a = ap.parse_args(); L = a.length
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from candidate import load
if a.default_entry:
    from candidate import SELECTED_OUTPUT_ARTIFACT
    assert a.selected and a.front and a.front_artifact=='front8' and a.out_artifact is None
    candidate=load()
    a.out_artifact=SELECTED_OUTPUT_ARTIFACT
else:
    candidate = load(None if a.selected else a.artifact, fuse_front=a.front, front_artifact=a.front_artifact,out_artifact=a.out_artifact)
ext=extension(a.artifact)
if a.front_artifact=='front8' and hasattr(ext,'bias_head_last') and ext.bias_head_last():
    a.front_artifact='front8_hlast'

report = dict(artifact=a.artifact,length=L,baseline='installed TriangleAttention.eval() under inference_mode',
              out_artifact=a.out_artifact,actual_default_entry=a.default_entry,
              selected_entry=a.selected,
              rounds=a.rounds,replays=a.replays,records=[],inference_front=a.front,
              build=json.loads((Path(__file__).parent/a.artifact/'build-ready.json').read_text()))
if a.out_artifact:report['output_build']=json.loads((Path(__file__).parent/a.out_artifact/'build-ready.json').read_text())
if a.front:
    report['front_artifact'] = a.front_artifact
    report['front_build'] = json.loads((Path(__file__).parent/a.front_artifact/'build-ready.json').read_text())
def save(): a.output.write_text(json.dumps(report,indent=2)+'\n')

def sampled_fp64(model,x,mask,baseline,cand):
    xx = x if model.starting else x.transpose(1,2).contiguous()
    z = F.layer_norm(xx.double(),(128,),model.ln_pair.weight.double(),model.ln_pair.bias.double(),
                     model.ln_pair.eps).bfloat16().double()
    rows = torch.tensor([0,L//2,L-1],device=x.device)
    queries = torch.tensor(sorted(set([0,1,L//3,L//2,L-1])),device=x.device)
    q,k,v,g = [F.linear(z[:,rows],getattr(model,n).weight.double()).bfloat16().double()
               .view(1,len(rows),L,4,32).permute(0,3,1,2,4)
               for n in ('to_query','to_key','to_value','to_gate')]
    b = F.linear(z[:,queries],model.to_bias.weight.double()).bfloat16().double().permute(0,3,1,2)
    logits = q[:,:,:,queries]@k.transpose(-1,-2)/32**.5+b[:,:,None]
    if mask is not None: logits.masked_fill_(~mask[:,None,None,None,:],-torch.inf)
    o = (logits.softmax(-1).nan_to_num(0.)@v).bfloat16().double()
    gated = (o*g[:,:,:,queries].sigmoid()).bfloat16().double().permute(0,2,3,1,4).reshape(1,len(rows),len(queries),128)
    update = F.linear(gated,model.to_out.weight.double()).bfloat16()
    sample_x = xx[:,rows][:,:,queries]
    exact = sample_x+update
    def sample(y):
        yy = y if model.starting else y.transpose(1,2)
        return yy[:,rows][:,:,queries]
    base_error = rel(sample(baseline),exact);cand_error = rel(sample(cand),exact)
    assert cand_error < .003 and cand_error <= base_error*1.15+.0004,(base_error,cand_error)
    return dict(baseline_relative=base_error,candidate_relative=cand_error,
                update_rms=float(update.float().square().mean().sqrt()))

for ending in (False,True):
    torch.manual_seed(92638)
    model = TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                              implementation='triton',p_drop=.25).cuda().bfloat16().eval()
    with torch.no_grad():
        for name,w in model.named_parameters():
            if w.ndim>=2: w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'): w.fill_(1)
            else: w.zero_()
    x = torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    mask = torch.ones(1,L,device='cuda',dtype=torch.bool)
    mask[:,::7] = False
    with torch.inference_mode():
        cases = ('mixed','dense','one_key','all_masked','none') if a.qualify else ('mixed',)
        for case in cases:
            mask.fill_(True)
            if case == 'mixed': mask[:,::7] = False
            elif case in ('one_key','all_masked'):
                mask.zero_()
                if case == 'one_key': mask[:,L-3] = True
            active_mask = None if case == 'none' else mask
            ref = model(x,active_mask); got = candidate(model,x,active_mask)
            assert torch.isfinite(got).all()
            output_rel = rel(got,ref)
            # The residual must not conceal errors in the attention contribution.
            attention_rel = rel(got.float()-x.float(),ref.float()-x.float())
            assert output_rel < .003 and attention_rel < .025,(ending,case,output_rel,attention_rel)
            if case == 'all_masked': assert torch.equal(got,x)
            report['records'].append(dict(kind='correctness',ending=ending,case=case,
                                          output_relative=output_rel,attention_relative=attention_rel))
            if a.qualify and case in ('mixed','one_key','all_masked'):
                metrics = sampled_fp64(model,x,active_mask,ref,got)
                if case != 'all_masked': assert metrics['update_rms'] > 1e-3
                report['records'].append(dict(kind='fp64_full_module_sample',ending=ending,case=case,**metrics))
            save()
        mask.fill_(True); mask[:,::7] = False
        if a.profile:
            for name,fn in [('baseline',lambda:model(x,mask)),('candidate',lambda:candidate(model,x,mask))]:
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],acc_events=True) as prof:
                    for _ in range(3):
                        yy = fn(); torch.cuda.synchronize()
                events = [dict(kernel=e.name,us=e.device_time_total) for e in prof.events()
                          if e.device_type == torch.autograd.DeviceType.CUDA]
                report['records'].append(dict(kind='cuda_profile',ending=ending,path=name,events=events)); save()
                del yy
        if a.profile_only: continue
        timing,graphs = paired([lambda:model(x,mask),lambda:candidate(model,x,mask)],a.rounds,a.replays)
        report['records'].append(dict(kind='paired_module',ending=ending,**timing)); save()
        print('INFERENCE_MODULE',L,ending,timing,flush=True)
        x.mul_(.8); model.to_query.weight.add_(.001); model.to_gate.weight.sub_(.002)
        mask[:,::5] = False
        graphs[0][0].replay(); graphs[1][0].replay(); torch.cuda.synchronize()
        changed_rel = rel(graphs[1][1],graphs[0][1])
        assert changed_rel < .003,changed_rel
        report['records'].append(dict(kind='changed_input_weight_mask_graph',ending=ending,relative=changed_rel))
        del graphs
        if a.qualify:
            compiled = torch.compile(lambda p,m:candidate(model,p,m),fullgraph=True)
            y = compiled(x,mask)
            assert rel(y,model(x,mask)) < .003
            report['records'].append(dict(kind='fullgraph_compile',ending=ending,passed=True))
    with torch.no_grad():
        assert rel(candidate(model,x,mask),model(x,mask)) < .003
    report['records'].append(dict(kind='no_grad',ending=ending,passed=True)); save()
report['complete'] = True; save()
