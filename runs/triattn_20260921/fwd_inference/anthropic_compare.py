"""Same-process paired complete inference FWD: pristine Anthropic release versus our kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent/'oc-release'
sys.path.insert(0,str(RELEASE))
import opt_core
from opt_core.attn import pair_fused as pf
assert Path(opt_core.__file__).resolve().parent == RELEASE/'opt_core'

import torch
from miniworld_engine.modules import TriangleAttention
from candidate import load
from check import paired, rel

ap = argparse.ArgumentParser()
ap.add_argument('--length',required=True,type=int)
ap.add_argument('--output',required=True,type=Path)
ap.add_argument('--rounds',type=int,default=64)
ap.add_argument('--replays',type=int,default=40)
ap.add_argument('--core',default='default')
ap.add_argument('--artifact',default='resident6')
ap.add_argument('--previous',action='store_true')
ap.add_argument('--previous-artifact',default='resident6')
ap.add_argument('--selected',action='store_true')
ap.add_argument('--out-artifact')
ap.add_argument('--previous-out-artifact')
ap.add_argument('--default-entry',action='store_true')
a = ap.parse_args(); L = a.length
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
if a.default_entry:
    from candidate import SELECTED_OUTPUT_ARTIFACT
    assert a.selected and a.out_artifact is None
    ours=load()
    a.out_artifact=SELECTED_OUTPUT_ARTIFACT
else:
    ours = load(None if a.selected else a.artifact,out_artifact=a.out_artifact)
previous = load(None if a.previous_artifact=='selected' else a.previous_artifact,out_artifact=a.previous_out_artifact) if a.previous and (a.artifact!=a.previous_artifact or a.out_artifact!=a.previous_out_artifact) else None
report = dict(length=L,release_root=str(RELEASE),core=a.core,artifact=a.artifact,selected_entry=a.selected,records=[],
              out_artifact=a.out_artifact,previous_out_artifact=a.previous_out_artifact,actual_default_entry=a.default_entry,
              rounds=a.rounds,replays=a.replays,scope='complete inference FWD with out-of-place residual',
              affine='nontrivial BF16-representable values in FP32 parameters; matches upstream internal affine rounding',
              release_source_sha256={},candidate_builds={})
original = Path('/home/psk6950/ext/uplifting-biomolecular-modeling/common/opt_core/opt_core')
for name in ('attn/pair_fused.py','attn/pair_fused_cells.json','kernels/triattn/cuda_sm90a/__init__.py',
             'kernels/triattn/cuda_sm90a/csrc/triattn_mw.cu'):
    digest = hashlib.sha256((RELEASE/'opt_core'/name).read_bytes()).hexdigest()
    assert digest == hashlib.sha256((original/name).read_bytes()).hexdigest()
    report['release_source_sha256'][name] = digest
from candidate import SELECTED_ARTIFACTS
from native import extension
actual=SELECTED_ARTIFACTS[L] if a.selected else a.artifact
ext=extension(actual)
front_name='front8_hlast' if hasattr(ext,'bias_head_last') and ext.bias_head_last() else 'front8'
for name in (actual,front_name):
    report['candidate_builds'][name] = json.loads((ROOT/name/'build-ready.json').read_text())
if a.out_artifact:report['candidate_builds'][a.out_artifact]=json.loads((ROOT/a.out_artifact/'build-ready.json').read_text())
if previous is not None:
    prev_name=SELECTED_ARTIFACTS[L] if a.previous_artifact=='selected' else a.previous_artifact
    report['previous_build']=json.loads((ROOT/prev_name/'build-ready.json').read_text())
def save(): a.output.write_text(json.dumps(report,indent=2,default=str)+'\n')

for ending in (False,True):
    torch.manual_seed(92638)
    model = TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                              implementation='triton',p_drop=.25).cuda().bfloat16().eval()
    with torch.no_grad():
        for name,w in model.named_parameters():
            if w.ndim>=2: w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'): w.normal_(mean=1,std=.05)
            else: w.normal_(std=.05)
            if w.ndim==1: w.copy_(w.bfloat16().float())
    x = torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
    mask = torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    m5 = mask[:,None,None,None,:].expand(-1,L,-1,-1,-1)
    with torch.inference_mode():
        weights = dict(w_o=model.to_out.weight,w_q=model.to_query.weight,w_k=model.to_key.weight,
                       w_v=model.to_value.weight,w_g=model.to_gate.weight,w_b=model.to_bias.weight,
                       ln_w=model.ln_pair.weight,ln_b=model.ln_pair.bias,n_heads=4,head_dim=32,eps=model.ln_pair.eps)
        packed = pf.pack_triattn_weights(**weights)
        kw = dict(ending=ending,impl='fpf',core=a.core,ln='fused')
        plan = pf._plan_triattn(x,packed,m5,residual=False,variant=None,engage_cells=False,**kw)
        report['records'].append(dict(kind='release_plan',ending=ending,plan=plan));save()
        def release_add():
            return x+pf.tri_attn_block(x,packed,m5,residual=False,**kw)
        work = torch.empty_like(x)
        def release_fused():
            work.copy_(x)
            return pf.tri_attn_block(work,packed,m5,residual=True,**kw)
        def candidate(): return ours(model,x,mask)
        for case in ('mixed','dense','one_key','all_masked'):
            mask.fill_(True)
            if case == 'mixed':mask[:,::7]=False
            elif case in ('one_key','all_masked'):
                mask.zero_()
                if case == 'one_key':mask[:,L-3]=True
            base = release_add().clone(); fused_base = release_fused().clone(); got = candidate()
            torch.cuda.synchronize()
            assert all(torch.isfinite(t).all() for t in (base,fused_base,got))
            output_rel = rel(got,base)
            update_rel = rel(got.float()-x.float(),base.float()-x.float())
            rec = dict(kind='correctness',ending=ending,case=case,output_relative=output_rel,
                       update_relative=update_rel,release_forms_relative=rel(base,fused_base))
            if case != 'all_masked':
                assert output_rel<.003 and update_rel<.025,rec
                assert rec['release_forms_relative']<.003,rec
            else:
                rec['ours_equals_input'] = torch.equal(got,x)
                rec['release_equals_input'] = torch.equal(base,x)
                rec['semantic_difference'] = not rec['release_equals_input'] and rec['ours_equals_input']
            report['records'].append(rec);save()
        mask.fill_(True);mask[:,::7]=False
        for form,baseline in [('update_plus_add',release_add),('copy_plus_fused_residual',release_fused)]:
            timing,graphs = paired([baseline,candidate],a.rounds,a.replays)
            assert rel(graphs[1][1],graphs[0][1])<.003
            report['records'].append(dict(kind='paired_module',ending=ending,release_form=form,**timing));save()
            print('ANTHROPIC_PAIRED',L,ending,form,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
            del graphs
        if previous is not None:
            timing,graphs=paired([lambda:previous(model,x,mask),candidate],a.rounds,a.replays)
            assert rel(graphs[1][1],graphs[0][1])<.003
            report['records'].append(dict(kind='paired_previous',ending=ending,previous=a.previous_artifact,**timing));save()
            print('PREVIOUS_PAIRED',L,ending,{k:v for k,v in timing.items() if k!='paired_ratios'},flush=True)
            del graphs
        x.mul_(.8);mask[:,::5]=False
        assert rel(candidate(),release_add())<.003
        for name,fn in [('anthropic_update_plus_add',release_add),('anthropic_copy_plus_fused',release_fused),('ours',candidate)]:
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],acc_events=True) as prof:
                for _ in range(3):
                    out = fn();torch.cuda.synchronize()
            events = [dict(kernel=e.name,us=e.device_time_total) for e in prof.events()
                      if e.device_type==torch.autograd.DeviceType.CUDA]
            report['records'].append(dict(kind='profile',ending=ending,path=name,events=events));save()
        report['served_cores'] = dict(pf.SERVED_CORES)
        report['tier_memo'] = {str(k):repr(v) for k,v in pf._TIER_CORE_MEMO.items()}
        from opt_core.kernels.triattn import cuda_sm90a
        report['native_release_install'] = cuda_sm90a._INSTALL_REPORT
        save()
report['imported_release_sources'] = {}
for name,mod in list(sys.modules.items()):
    if name=='opt_core' or name.startswith('opt_core.'):
        path = getattr(mod,'__file__',None)
        if path:
            pp = Path(path).resolve();assert str(pp).startswith(str(RELEASE.resolve())+'/'),(name,path)
            report['imported_release_sources'][str(pp.relative_to(RELEASE))] = hashlib.sha256(pp.read_bytes()).hexdigest()
report['complete'] = True;save()
