"""Generate and compile the policy variant without creating a CUDA context."""
from pathlib import Path
from types import SimpleNamespace
import sys,subprocess
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r.parents[1]/'.engine-release-2.0.0/src'))
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
prior=SimpleNamespace()
scope=dict(Path=Path,__file__=str(r/'wide_two_group_contract_gp.py'),self=prior,channel_group=1)
s=(r/'wide_two_group_contract_gp.py').read_text();a=s.index('        root=Path(');b=s.index('        flags=',a)
exec('if True:\n'+s[a:b],scope)
p=SimpleNamespace(D=512,n=384);prior.p=p;prior.smem=98304+128
plan=SimpleNamespace(p=p,contract_gp=prior)
from wide_mask_transform import mask_stage
for policy in ('first','last'):
    scope=dict(plan=plan,self=SimpleNamespace(),policy=policy,mask_stage=mask_stage)
    s=(r/'gp_store_policy.py').read_text();a=s.index('        p=plan.p;');b=s.index('        flags=',a)
    exec('if True:\n'+s[a:b],scope)
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=512']
    cubin=T.compile_text(scope['body'],flags);print(policy,cubin,flush=True)
    subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
