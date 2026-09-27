"""Compile generated contraction sources on an allocated CPU, without CUDA init."""
from pathlib import Path
from types import SimpleNamespace
import sys,subprocess,os
THIS=Path(__file__).resolve().parent
os.environ['MINIWORLD_ENGINE_JIT_ROOT']=str(THIS/'engine_cache')
sys.path.insert(0,str(THIS.parent.parent/'.engine-release-2.0.0/src'))
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_n96_contract_gp import source
for d in (384,512):
    obj=SimpleNamespace(source_text=source())
    scope=dict(self=obj,unroll=True)
    script=(THIS/'wide_m192n96_contract_gp.py').read_text()
    begin=script.index('        body=self.source_text');end=script.index('        self.source_text=body;p=',begin)
    exec('if True:\n'+script[begin:end],scope)
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}','-DFIXED_LENGTH=768']
    cubin=T.compile_text(scope['body'],flags)
    print('CUBIN',d,str(cubin),flush=True)
    subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
