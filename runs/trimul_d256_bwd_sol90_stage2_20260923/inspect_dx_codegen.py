"""CPU-only compiler inspection; run on an allocated compute node."""
import os,sys,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent.parent/'.engine-release-2.0.0/src'))
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
cache=R/'engine_cache/trimul_h100'
for name,source in (
 ('pipe','d85d9a632d33dee8ee434670f88b96cf0886a223b89f00e89450ea27484d8e76'),
 ('pipe-unroll','52f0579f665ba43b2d76c097ab6eda86410ea608899b1f0a50b61e7ccd56edc7'),
):
 out=R/f'dx-{name}.ptx'
 subprocess.run(['/usr/local/cuda-12.9/bin/nvcc','-std=c++17','-O3','-arch=sm_90a','--ptx','-lineinfo','-I'+str(T._upstream()/'csrc'),'-DWEIGHT_SPLITS=8',str(cache/(source+'.cu')),'-o',str(out)],check=True)
 lines=out.read_text().splitlines()
 print(name, 'PTX_LINES',len(lines),flush=True)
 for center in ((933,1122,1287,1654) if name=='pipe' else (902,1060,1251)):
  print('\n'.join(f'{i+1}: {lines[i]}' for i in range(max(0,center-10),min(len(lines),center+8))),flush=True)
