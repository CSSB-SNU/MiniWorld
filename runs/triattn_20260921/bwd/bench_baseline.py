"""TriangleAttention backward baseline: actual core and complete module.

No parameter updates or gradient accumulation; all gradients are returned by
autograd.grad. Dropout disabled for kernel attribution. All output weights are
nonzero. Timed graphs are warmed and include every backward preparation/reduce.
"""
import argparse,gc,hashlib,json,statistics
from pathlib import Path
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key

ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
L=a.length
torch.manual_seed(92301)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
report=dict(length=L,width=128,heads=4,head_dim=32,batch=1,dropout=0,torch=torch.__version__,device=torch.cuda.get_device_name(),source=core.__file__,source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),measurements={})

def save():a.output.write_text(json.dumps(report,indent=2)+'\n')

def capture(fn,stream=None):
    s=stream or torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):out=fn()
    torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize();del out
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s):out=fn()
    return g,out

def measure(name,fn,stream=None):
    print('WARM',name,flush=True)
    g,out=capture(fn,stream)
    g.replay();torch.cuda.synchronize()
    values=out if isinstance(out,(tuple,list)) else (out,)
    assert all(torch.isfinite(v).all() for v in values),name
    norms=[float(v.float().norm()) for v in values]
    assert any(v>0 for v in norms),name
    times=[]
    for _ in range(7):
        st,en=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        st.record()
        for j in range(20):g.replay()
        en.record();en.synchronize();times.append(st.elapsed_time(en)*1000/20)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(3):g.replay()
        torch.cuda.synchronize()
    kernels={}
    for e in p.events():
        if e.device_type==torch.autograd.DeviceType.CUDA:
            d=kernels.setdefault(e.name,dict(us=0.,calls=0))
            d['us']+=e.device_time_total/3;d['calls']+=1
    result=dict(graph_us=statistics.median(times),rounds_us=times,kernels=sorted([dict(name=k,**v) for k,v in kernels.items()],key=lambda x:-x['us']),output_norms=norms)
    report['measurements'][name]=result;save();print('MEASURED',name,result['graph_us'],'kernels',result['kernels'][:8],flush=True)
    del out,values,g;p=None;gc.collect();torch.cuda.empty_cache()

q,k,v=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(3)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
b[...,::7]=torch.finfo(b.dtype).min
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
dy=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4)
def core_bwd():
    dq,dk,dv,db=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))
    return dq,dk,dv,db.reshape(1,4,L,L,L).sum(2)
measure('core_backward',core_bwd)
report['dbias_raw_bytes']=4*L**3*2;save()
del q,k,v,b,out,m,dy;gc.collect();torch.cuda.empty_cache()

for ending in (False,True):
    model=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    model._fuse_projection_backward=False
    model._fuse_bias_backward=False
    with torch.no_grad():
        for name,w in model.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    dy=torch.randn_like(x);parameters=(x,*model.parameters())
    backward_stream=torch.cuda.Stream();backward_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(backward_stream):y=model(x,mask)
    torch.cuda.current_stream().wait_stream(backward_stream)
    def backward():return torch.autograd.grad(y,parameters,dy,retain_graph=True)
    measure('module_backward_e%d'%ending,backward,backward_stream)
    def forward_backward():
        z=model(x,mask)
        return torch.autograd.grad(z,parameters,dy)
    measure('module_forward_backward_e%d'%ending,forward_backward)
    del y,x,dy,parameters,model;gc.collect();torch.cuda.empty_cache()
report['complete']=True;save();print('COMPLETE',a.output,flush=True)
