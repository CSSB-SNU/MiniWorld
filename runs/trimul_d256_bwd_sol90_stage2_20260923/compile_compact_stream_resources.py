from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.engine-release-2.0.0/src'))
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
r=Path(__file__).resolve().parent
body=(r/'wide_compact_stream_contract_gp.cu').read_text().replace('// MMA_HELPERS',(r.parent/'trimul_d256_bwd_sol90_20260923/mma.cuh').read_text())
body=body.replace('p.N','384').replace('  for(int it=0;it<steps;++it)', '  #pragma unroll 1\n  for(int it=0;it<steps;++it)')
flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=512','-DSTREAM_MINBLOCKS=2','-DSTORE_READ_CREDIT=1']
path=T.compile_text(body,flags)
print(path,flush=True)
import subprocess
print(subprocess.check_output(['cuobjdump','-res-usage',str(path)],text=True),flush=True)
