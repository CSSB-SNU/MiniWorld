from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent
class LN:
 def __init__(self,p):
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
  body=(R/'ln.cu').read_text()
  self.threads=int(os.environ.get('LN_THREADS','256'))
  minblocks=int(os.environ.get('LN_MINBLOCKS','2'))
  assert self.threads in (128,256) and minblocks in (1,2,3)
  body=body.replace('NT=256','NT='+str(self.threads)).replace('__launch_bounds__(NT,2)',f'__launch_bounds__(NT,{minblocks})')
  body=body.replace('kc+=2','kc+=NT/128').replace('r+=8','r+=NT/32')
  if os.environ.get('LN_AGG')=='1':
   body=body.replace('extern "C" __global__',(R/'ln_aggregate.cuh').read_text()+'\nextern "C" __global__')
   body=body.replace(' for(int q=0;q<16;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}',' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);')
  if os.environ.get('LN_CACHE')=='1':
   body=body.replace('z[16],v[16],s0','z[16],v[16],cache_dy[16],s0')
   body=body.replace('v[q]=dy*p.gamma[c];','cache_dy[q]=dy;v[q]=dy*p.gamma[c];')
   body=body.replace('float dy=__bfloat162float(p.dn[size_t(row+r)*H+c]);float centered','float dy=cache_dy[q];float centered')
  self.smem=65664
  if os.environ.get('LN_STATS_TMA')=='1':
   body=body.replace('mbar_arrive_expect_tx(bar,SB);','''mbar_arrive_expect_tx(bar,SB+512);
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+SB+128)),"l"(p.mu+row),"n"(256),"r"(smem_u32(bar)):"memory");
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+SB+384)),"l"(p.rs+row),"n"(256),"r"(smem_u32(bar)):"memory");''')
   body=body.replace('float mu=p.mu[row+r],rs=p.rs[row+r],','float mu=reinterpret_cast<float*>(sm+SB+128)[r],rs=reinterpret_cast<float*>(sm+SB+384)[r],')
   self.smem+=512
  if os.environ.get('LN_STORE_READ')=='1':
   body=body.replace('tma_store_commit();tma_store_wait_all();','tma_store_commit();tma_store_wait_read<0>();')
   end=body.rfind('\n }\n')
   assert end>=0
   body=body[:end+4]+' if(tid==0)tma_store_wait_all();__syncthreads();\n'+body[end+4:]
  if os.environ.get('LN_GAMMA_SMEM')=='1':
   # Preserve the barrier and optional stats staging area; still three CTAs/SM.
   body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+SB+1024)[c]')
   hook=' if(tid==0){mbar_init(bar,1);fence_barrier_init();}'
   assert body.count(hook)==1
   body=body.replace(hook,' for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+SB+1024)[c]=p.gamma[c];\n'+hook)
   self.smem=65536+1024+512*4
  out=T.compile_text(body,flags);self.cubin=out;self.k=T.load_unit(str(out),'mw_d256_ln').kernel('mw_d256_ln');self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();tm=lambda t:L.tensor_map(t,[64,64],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  self.params=L.Struct([tm(p.tri),tm(p.dt),p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
  drv=self.k.unit.drv
  occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),self.threads,self.smem)))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*min(minblocks,occ)
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
