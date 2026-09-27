"""Compile the same generated sources without creating a CUDA context."""
from pathlib import Path
from types import SimpleNamespace
import sys,subprocess,os
THIS=Path(__file__).resolve().parent
os.environ['MINIWORLD_ENGINE_JIT_ROOT']=str(THIS/'engine_cache')
sys.path.insert(0,str(THIS.parent.parent/'.engine-release-2.0.0/src'))
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
for width in (512,):
    obj=SimpleNamespace();p=SimpleNamespace(D=width,n=768)
    scope=dict(Path=Path,__file__=str(THIS/'wide_two_group_contract_gp.py'),self=obj,channel_group=2 if width==384 else 1)
    source=(THIS/'wide_two_group_contract_gp.py').read_text()
    a=source.index('        root=Path(');b=source.index('        flags=',a)
    exec('if True:\n'+source[a:b],scope)
    original=obj.source_text;plan=SimpleNamespace(p=p,contract_gp=SimpleNamespace(source_text=original))
    scope.update(prior=plan.contract_gp,plan=plan,p=p,n=768)
    source=(THIS/'wide_bf12_contract_gp.py').read_text()
    a=source.index('        body=prior.source_text');b=source.index('        flags=',a)
    exec('if True:\n'+source[a:b],scope)
    source=(THIS/'wide_ptx_bf12_contract_gp.py').read_text();scope['body']=obj.source_text
    a=source.index('        begin=body.index(');b=source.index('        flags=',a)
    exec('if True:\n'+source[a:b],scope)
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={width}']
    cubin=T.compile_text(scope['body'],flags)
    print('CUBIN',width,str(cubin),flush=True)
    subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
