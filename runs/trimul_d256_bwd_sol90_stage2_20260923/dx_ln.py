from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class DxLN:
 def __init__(self,p,splits):
  body=(PRE/'wide_finish.cu').read_text().split('// Per-warp LN.')[0].replace('int M,L;','int M,L;CUtensorMap xmap,resmap,dxmap;')
  body+='\n'+(R/'ln_aggregate.cuh').read_text()+'\n'+(R/'dx_ln.cuh').read_text()
  if os.environ.get('DX_WARP')=='1':body=(R/'dx_ln_warp.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
  self.threads=256
  if os.environ.get('DX_SPLIT','0')!='0':
   assert os.environ.get('DX_WARP')=='1' and os.environ.get('DX_N256','0')=='0' and os.environ.get('DX_PIPE','0')=='0'
   body=(R/'dx_split.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text());self.threads=384
   if os.environ['DX_SPLIT']=='2':
    body=body.replace('template<int WG> TMN_DEVI void consumer','TMN_DEVI void consumer')
    body=body.replace('constexpr int SYNC=WG+1;int tid=', 'int WG=threadIdx.x/128-1,SYNC=WG+1;int tid=')
    body=body.replace('if(tid<256)consumer<0>(p,sm,bars);else consumer<1>(p,sm,bars);','consumer(p,sm,bars);')
   else:assert os.environ['DX_SPLIT']=='1'
  if os.environ.get('DX_PIPE','0')!='0':
   assert os.environ.get('DX_WARP')=='1' and os.environ.get('DX_N256','0')=='0'
   from dx_pipeline import pipeline
   body=pipeline(body,int(os.environ['DX_PIPE']))
  if os.environ.get('DX_N256')=='1':
   assert os.environ.get('DX_WARP')=='1'
   from dx_n256 import widen
   body=widen(body)
  if os.environ.get('DX_NO_DW')=='1':
   assert os.environ.get('DX_WARP')=='1'
   start=body.index(' for(int which=0;which<4;++which)');end=body.index('extern "C" __global__',start)
   body=body[:start]+'}\n'+body[end:]
  if os.environ.get('DX_IN_STATS')=='1':
   assert os.environ.get('DX_WARP')=='1' and hasattr(p,'input_stats')
   import re
   body,count=re.subn(r'float z=0;for\(int c=lane;c<D;c\+=32\)z\+=rd\(x,pos\(r,c\)\);float mu=wsum\(z\)/D;z=0;\s*for\(int c=lane;c<D;c\+=32\)\{float val=rd\(x,pos\(r,c\)\)-mu;z\+=val\*val;\}float rs=rsqrtf\(wsum\(z\)/D\+1e-5f\),s0=0,s1=0;', 'float mu=p.f[12][2*(row+r)],rs=p.f[12][2*(row+r)+1],s0=0,s1=0;',body)
   assert count==1,count
  gamma_cache=os.environ.get('DX_GAMMA_CACHE')=='1'
  if gamma_cache:
   assert os.environ.get('DX_WARP')=='1'
   body=body.replace('p.f[0][c]','reinterpret_cast<float*>(sm+98432)[c]')
   hook=' __syncthreads();cooperative_groups::this_grid().sync();'
   assert body.count(hook)==1
   body=body.replace(hook,' for(int c=tid;c<D;c+=blockDim.x)reinterpret_cast<float*>(sm+98432)[c]=p.f[0][c];\n'+hook)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256','-DWIDTH_N=128','-DWIDTH_GROUPS=2',f'-DWEIGHT_SPLITS={splits}','-DFUSED_GP=0']
  out=T.compile_text(body,flags);self.cubin=out;self.k=T.load_unit(str(out),'mw_d256_dx_ln').kernel('mw_d256_dx_ln');self.smem=99456 if gamma_cache else 98432;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();t7=p.tensors.copy();t7[22:24]=[p.dl,p.dr]
  ff=p.floats.copy()
  if os.environ.get('DX_IN_STATS')=='1':ff[12]=p.input_stats
  self.params=L.Struct([*p.maps7,*t7,*ff,p.M,p.n,W.tm(p.x.reshape(p.M,256)),W.tm(p.dy.reshape(p.M,256)),W.tm(p.dx.reshape(p.M,256))])
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),self.threads,self.smem)));self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
  if self.threads==384:
   nr=int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(drv.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,drv.d.CUfunction(int(self.k.handle)))))
   assert self.threads*nr >= 128*(32+104+104),f'insufficient per-CTA register pool: {nr}'
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
