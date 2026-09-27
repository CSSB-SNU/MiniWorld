"""Publish derivative tiles with TMA while their local dW WGMMA executes."""
from pathlib import Path
import os,re
import torch
from b7 import B7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
PRE=Path(__file__).resolve().parent.parent/'trimul_d256_bwd_sol90_20260923'
class TmaB7(B7):
 def __init__(self,p,leaves,packed=None,mask=None,splits=None):
  super().__init__(p,splits if splits is not None else 8 if p.n==384 else 16)
  if mask is not None:self.mask=mask
  body=(PRE/'source.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text()).replace('mw_d256_b7_source','mw_d256_b7_tma')
  body=body.replace('CUtensorMap xn,w,dl,dr;','CUtensorMap xn,w,dl,dr,gmap[4];')
  body=body.replace('   p.gp[side*2][size_t(col)*p.M+row+r]=dp;','').replace('   p.gp[side*2+1][size_t(col)*p.M+row+r]=dg;','')
  helper='''
TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
'''
  body=body.replace('extern "C" __global__',helper+'\nextern "C" __global__')
  if os.environ.get('GP_EVICT')=='1':
   body=body.replace('cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];','{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],pol;}')
  if packed if packed is not None else os.environ.get('GP_PACK')=='1':
   body=body.replace('extern "C" __global__',(Path(__file__).with_name('packed_glu.cuh')).read_text()+'\nextern "C" __global__')
   body=re.sub(r'  // Shared derivative matrix.*?  __syncthreads\(\);fence_proxy_async\(\);__syncthreads\(\);', '''  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_glu(pre,xn,sm+DERIV,ma,mb);
  __syncthreads();fence_proxy_async();__syncthreads();''',body,flags=re.S)
  body=body.replace('  fence_regs(dw0);fence_regs(dw1);wgmma_fence();','''  if(tid==0){store_gp(p.gmap+(rank/16)*2,sm+DERIV+4096,row,(rank%16)*32);store_gp(p.gmap+(rank/16)*2+1,sm+DERIV,row,(rank%16)*32);tma_store_commit();}
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();''')
  body=body.replace('wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);__syncthreads();','wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();__syncthreads();')
  if os.environ.get('GP_OFF')=='1':
   body=body.replace('extern "C" __global__',(Path(__file__).with_name('mma_offset.cuh')).read_text()+'\nextern "C" __global__')
   body=body.replace('mma64<0,0>(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+(k/4)*8192+(k%4)*32),16,1024,1),k>0);','mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);')
   body=body.replace('uint64_t a=smem_desc(smem_u32(sm+DERIV+k*32),16,1024,1);','uint64_t a=smem_desc(smem_u32(sm+DERIV),16,1024,1);')
   body=body.replace('mma128<0,1>(dw0,a,smem_desc(smem_u32(xn+k*2048),8192,1024,1),it>0||k>0);','mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);')
   body=body.replace('mma128<0,1>(dw1,a,smem_desc(smem_u32(xn+16384+k*2048),8192,1024,1),it>0||k>0);','mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);')
  self.source_threads=128
  if os.environ.get('GP_WARP')=='1':
   from warp_source import specialize
   body=specialize(body);self.source_threads=256
  self.source_text=body
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
  out=T.compile_text(body,flags);self.source=T.load_unit(str(out),'mw_d256_b7_tma').kernel('mw_d256_b7_tma');self.source.set_max_dynamic_smem(114816)
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[x.stride(0)*2],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),*[dy(x) for x in p.gp],self.mask,*p.gp,p.floats[7],p.M])
  if os.environ.get('LN_AGG')=='1':
   base=(PRE/'base.cuh').read_text().replace('template<bool INPUT> TMN_DEVI void ln_bwd(const Params& p){',(Path(__file__).with_name('ln_aggregate.cuh')).read_text()+'\ntemplate<bool INPUT> TMN_DEVI void ln_bwd(const Params& p,uint8_t* sm){')
   base=base.replace(' for(int c=lane;c<C;c+=32){atomicAdd(p.f[INPUT?8:10]+c,gg[c/32]);atomicAdd(p.f[INPUT?9:11]+c,bb[c/32]);}',' aggregate_ln<C,THREADS>(gg,bb,reinterpret_cast<float*>(sm),p.f[INPUT?8:10],p.f[INPUT?9:11]);')
   body=(PRE/'finish.cu').read_text().replace('#include "base.cuh"',base).replace('ln_bwd<true>(p)','ln_bwd<true>(p,sm)').replace('mw_d256_b7_finish','mw_d256_b7_finish_agg')
   out=T.compile_text(body,flags+['-DWIDTH=256']);self.finish=T.load_unit(str(out),'mw_d256_b7_finish_agg').kernel('mw_d256_b7_finish_agg');self.finish.set_max_dynamic_smem(W.tuning(256)[2])
   drv=self.finish.unit.drv
   occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)),256,W.tuning(256)[2])))
   self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 def __call__(self):
  self.source.launch((32*self.splits,1,1),(self.source_threads,1,1),[self.params],114816)
  if hasattr(self,'wide_finish'):self.wide_finish()
  else:W.launch(self.finish,self.p.params7,self.grid,D=256)
  return self.p.outputs
