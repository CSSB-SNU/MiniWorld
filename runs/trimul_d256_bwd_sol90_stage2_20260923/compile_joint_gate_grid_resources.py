"""Test whether power-of-two rank decoding restores spill-free allocation."""
from pathlib import Path
import sys,os,subprocess
THIS=Path(__file__).resolve().parent
sys.path.insert(0,str(THIS.parent.parent/'.engine-release-2.0.0/src'))
os.environ['MINIWORLD_ENGINE_JIT_ROOT']=str(THIS/'engine_cache')
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
for tag,name in [('v1','1e6e6e279fd6a72f0a1751ad3bec1a25e388b931173fa9eb7b01ba56d493f3ec'),('v2','b58e4f2a6705dcada6a00d12faffd73e378b4a15865ad9206d402033422f8ca9')]:
    body=(THIS/'engine_cache/trimul_h100'/f'{name}.cu').read_text()
    body=body.replace('blockIdx.x%36','(blockIdx.x%32+blockIdx.y*32)').replace('blockIdx.x/36','blockIdx.x/32')
    marker=' if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);'
    assert body.count(marker)==1
    body=body.replace(marker,' if(blockIdx.y && blockIdx.x%32>=4)return;\n'+marker)
    body=body.replace('mw_d256_joint_gate_source','mw_d256_joint_gate_grid')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWEIGHT_SPLITS=8']
    cubin=T.compile_text(body,flags);print('CUBIN',tag,str(cubin),flush=True)
    subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],check=True)
