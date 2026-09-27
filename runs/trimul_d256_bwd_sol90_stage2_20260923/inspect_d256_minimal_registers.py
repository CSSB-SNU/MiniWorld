"""Check exact compressed sigmoid correction and complete workload timing."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from d256_compact_four_source import CompactFourSource
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));producer,consumer=((56,88),(64,96),(48,88),(64,88))[index]
import subprocess
helper=(THIS/'mma_offset.cuh').read_text()
body='#include "tmn_kernels.cuh"\nusing namespace tmn;using namespace tmn::sm90;\n'+helper+'\nextern "C" __global__ __maxnreg__(LIMIT) void min_mma(float* out){\n extern __shared__ __align__(1024) uint8_t sm[];\n float acc[64]={};fence_regs(acc);wgmma_fence();\n mma128_off<0,0,0,1>(acc,0x4000004000010000ull,0x4000004002000200ull,0);\n wgmma_commit();wgmma_wait<0>();fence_regs(acc);\n static_for<64>([&](auto ii){constexpr int i=decltype(ii)::value;out[threadIdx.x*64+i]=acc[i];});\n}\n'
record={}
for limit in [80,88,90,96]:
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DLIMIT={limit}']
 try:
  cubin=T.compile_text(body,flags);record[limit]={'compiled':True,'cubin':str(cubin)}
 except Exception as ex:record[limit]={'compiled':False,'error':str(ex).split('ptxas fatal')[-2:]}
 print('MINIMAL',limit,record[limit],flush=True)
(THIS/f'result-d256-minimal-registers-{os.environ.get("SLURM_JOB_ID")}.json').write_text(json.dumps(record,indent=2))
