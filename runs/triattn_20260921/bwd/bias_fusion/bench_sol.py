from pathlib import Path
source=(Path(__file__).parent.parent/'bench_baseline.py').read_text().split('\nq,k,v=')[0]
exec(compile(source,str(Path(__file__).parent.parent/'bench_baseline.py'),'exec'))
import hybrid
import os
from native import extension,ARTIFACT_ROOT
extension()
report['native_build']=json.loads((ARTIFACT_ROOT/'build-ready.json').read_text())

q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
b[...,::7]=torch.finfo(b.dtype).min
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
def baseline():
    dq,dk,dv,db=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))
    return dq,dk,dv,db.reshape(1,4,L,L,L).sum(2)

ref=tuple(x.detach().clone() for x in baseline())
torch.cuda.synchronize()
report['errors']={}
row_options=[int(x) for x in os.environ.get('FUSION_ROW_OPTIONS','4 8').split()]
for rows in row_options:
    got=hybrid.backward(q,k,v,b,m,out,dy,rows)
    torch.cuda.synchronize()
    errors=[dict(relative_l2=float((x.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)),
        max_abs=float((x.float()-r.float()).abs().max()),finite=bool(torch.isfinite(x).all())) for x,r in zip(got,ref)]
    report['errors'][str(rows)]=errors;save();print('ERRORS',rows,errors,flush=True)
    assert all(e['finite'] and e['relative_l2']<.012 for e in errors),errors
    del got
del ref
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward as installed
assert extension() is not installed._extension(),'candidate and installed extension aliased'
report['installed_binary']=str(installed._extension().__file__)
report['candidate_binary']=str(extension().__file__)
measure('baseline',baseline)
measure('installed_group4',lambda:installed._backward(q,k,v,b,m,out,dy,True))
for rows in row_options:measure('group%d'%rows,lambda:hybrid.backward(q,k,v,b,m,out,dy,rows))
report['complete']=True;save()
