"""Compile only on an allocated compute node; never initialize or use a GPU."""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(R.parent.parent/'.engine-release-2.0.0/src'))
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from dx_n256 import widen
assert not torch.cuda.is_initialized()
common=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
warp=(R/'dx_ln_warp.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
rows=(R/'dx_ln_rows.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
packed=(R/'packed_glu.cuh').read_text().replace('packed_glu','packed_cluster_glu').replace('smem_u32(s+32768)','smem_u32(s)')
pairs=(R/'source_pairs.cu').read_text().replace('// MMA_HELPERS',(R/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',packed)
records=[]
for splits in (8,16):
 for name,body,flags in [
  ('n256',widen(warp),common+['-DWIDTH=256','-DWIDTH_N=128','-DWIDTH_GROUPS=2',f'-DWEIGHT_SPLITS={splits}','-DFUSED_GP=0']),
  ('rows128',rows,common+[f'-DWEIGHT_SPLITS={splits}']),
  ('rows256',widen(rows),common+[f'-DWEIGHT_SPLITS={splits}']),
  ('source_pairs',pairs,common+[f'-DWEIGHT_SPLITS={splits}']),
 ]:
  cubin=T.compile_text(body,flags)
  summary=[line for line in cubin.with_suffix('.ptxas.log').read_text().splitlines() if any(s in line for s in ('Function properties','stack frame','Used ','C7511','C7512'))]
  rec=dict(name=name,splits=splits,cubin=str(cubin),ptxas=summary);records.append(rec);print(rec,flush=True)
assert not torch.cuda.is_initialized()
(R/f'compile-shared-candidates-{os.environ.get("SLURM_JOB_ID")}.json').write_text(json.dumps(dict(complete=True,gpu_initialized=False,records=records),indent=2))
