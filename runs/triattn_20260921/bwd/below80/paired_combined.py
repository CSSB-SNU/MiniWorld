"""Paired current default module vs candidate; retain all five native fusions.

Extension selection occurs during warmup/capture; graph replay contains native
kernel launches and never depends on the subsequently restored Python variable.
"""
from pathlib import Path
source=(Path(__file__).parent.parent/'bench_baseline.py').read_text().split('\nq,k,v=')[0]
exec(compile(source,str(Path(__file__).parent.parent/'bench_baseline.py'),'exec'))
import copy,os,sys,importlib.util,hashlib
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward,dq_backward,ln_backward
ROOT=Path(__file__).resolve().parent
modules={'bias':bias_backward,'dq':dq_backward,'ln':ln_backward}
def load_extension(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);return ext
base_extensions={};candidate_extensions={}
report['baseline_binaries']={};report['candidate_binaries']={}
for kind,mod in modules.items():
    manifest=json.loads((ROOT/'baseline'/(kind+'_manifest.json')).read_text())
    base_extensions[kind]=load_extension(ROOT/'baseline'/manifest['binary'],manifest['module_name'])
    artifact=os.environ.get(kind.upper()+'_ARTIFACT','')
    if artifact:
        folder,lib={'bias':('bias_fusion','triattn_bias_fusion'),'dq':('dq','triattn_dq'),'ln':('ln_residual','triattn_ln_residual')}[kind]
        path=ROOT.parent/folder/artifact
        for filename,digest in json.loads((path/'build-ready.json').read_text()).items():
            assert hashlib.sha256((path/filename).read_bytes()).hexdigest()==digest
        name=(path/'module-name.txt').read_text().strip()
        candidate_extensions[kind]=load_extension(path/'build'/(lib+'.so'),name)
    else:
        candidate_extensions[kind]=mod._extension() if kind=='bias' else mod.extension()
    report['baseline_binaries'][kind]=base_extensions[kind].__file__
    report['candidate_binaries'][kind]=candidate_extensions[kind].__file__
assert any(candidate_extensions[k] is not base_extensions[k] for k in modules)

def timed(g):
    st,en=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    st.record()
    for _ in range(15):g.replay()
    en.record();en.synchronize()
    return st.elapsed_time(en)*1000/15

for ending in (False,True):
    torch.manual_seed(92301)
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    base._fuse_bias_backward=True
    assert all(getattr(base,n) for n in ('_fuse_projection_backward','_fuse_front_backward','_fuse_gate_backward','_fuse_dq_backward','_fuse_bias_backward'))
    cand=copy.deepcopy(base)
    x0=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    x1=x0.detach().clone().requires_grad_()
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    dy=torch.randn_like(x0)
    for regime in ('backward','forward_backward'):
        graphs=[];outs=[];contexts=[];ref=None
        for model,x in ((base,x0),(cand,x1)):
            for kind,mod in modules.items():mod._EXT=(base_extensions if model is base else candidate_extensions)[kind]
            s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            params=(x,*model.parameters())
            with torch.cuda.stream(s):y=model(x,mask)
            torch.cuda.current_stream().wait_stream(s)
            if regime=='backward':fn=lambda y=y,params=params:torch.autograd.grad(y,params,dy,retain_graph=True)
            else:fn=lambda model=model,x=x,params=params:torch.autograd.grad(model(x,mask),params,dy)
            g,out=capture(fn,s);g.replay();torch.cuda.synchronize()
            assert all(torch.isfinite(v).all() for v in out)
            if ref is None:ref=[v.detach().clone() for v in out]
            errs=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(out,ref)]
            assert max(errs)<.015,errs
            graphs.append(g);outs.append(out);contexts.append((s,y,params,fn))
        for kind,mod in modules.items():mod._EXT=base_extensions[kind]
        times=[[],[]];ratios=[]
        for rnd in range(12):
            pair={}
            for i in ((0,1) if rnd%2==0 else (1,0)):
                graphs[i].replay();pair[i]=timed(graphs[i]);times[i].append(pair[i])
            ratios.append(pair[0]/pair[1])
        result=dict(baseline_us=statistics.median(times[0]),candidate_us=statistics.median(times[1]),speedup=statistics.median(ratios),paired_ratios=ratios,rounds_us=times,gradient_relative_l2=errs)
        report['measurements']['%s_e%d'%(regime,ending)]=result;save()
        print('PAIRED',regime,ending,result['baseline_us'],result['candidate_us'],result['speedup'],flush=True)
        del graphs,outs,contexts,ref,g,out,s,y,params,fn;gc.collect();torch.cuda.empty_cache()
    del base,cand,x0,x1,dy,model,x;gc.collect();torch.cuda.empty_cache()
report['gate_fusion_enabled']=True
report['complete']=True;save()
