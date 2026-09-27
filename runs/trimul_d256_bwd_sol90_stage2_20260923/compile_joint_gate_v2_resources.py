"""Compile transformed D256 source on a compute CPU without initializing CUDA."""
from pathlib import Path
from types import SimpleNamespace
import sys,os,re,subprocess
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(THIS.parent.parent/'.engine-release-2.0.0/src'))
os.environ.update(MINIWORLD_ENGINE_JIT_ROOT=str(THIS/'engine_cache'),GP_OFF='1',GP_WARP='1')
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage
obj=SimpleNamespace(splits=8)
scope=dict(Path=Path,os=os,re=re,PRE=PRE,self=obj,packed=True,__file__=str(THIS/'tma_b7.py'))
s=(THIS/'tma_b7.py').read_text();a=s.index('  body=(PRE/');b=s.index('  self.source_text=body',a)
exec('if True:\n'+s[a:b],scope);obj.source_text=scope['body']
scope.update(original=SimpleNamespace(original=SimpleNamespace(source_text=obj.source_text)),mask_stage=mask_stage,__file__=str(THIS/'d256_joint_gate_source.py'))
s=(THIS/'d256_joint_gate_source.py').read_text();a=s.index('        body=mask_stage');b=s.index('        flags=',a)
exec('if True:\n'+s[a:b],scope)
scope['body']=obj.source_text
s=(THIS/'d256_joint_gate_source_v2.py').read_text();a=s.index('        begin=body.index(');b=s.index('        flags=',a)
exec('if True:\n'+s[a:b],scope)
flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWEIGHT_SPLITS=8']
cubin=T.compile_text(scope['body'],flags);print('CUBIN',str(cubin),flush=True)
subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
