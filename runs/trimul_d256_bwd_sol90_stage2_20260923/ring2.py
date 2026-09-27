"""Packed derivative ring with TMA publication and configurable consumer balance."""
from pathlib import Path
import ctypes,os,re,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class Ring2:
 def __init__(self,p,leaves):
  self.p=p;self.cohorts=int(os.environ.get('RCOHORTS','4'));self.consumers=int(os.environ.get('RCONSUMERS','32'));self.slots=int(os.environ.get('RSLOTS','32'))
  self.ring=p.x.new_empty((self.cohorts,self.slots,4,512,64));self.flags=torch.zeros((self.cohorts,self.slots,33),device=p.x.device,dtype=torch.int32);self.mask=p.mask.bfloat16()
  self.ctas=self.cohorts*(32+self.consumers)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.cohorts*4}','-DWIDTH=256']
  src=(PRE/'ring_source.cuh').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
  src=src.replace('CUtensorMap ringmap,dg','CUtensorMap ringmap,ringstore,dg').replace('CONSUMERS=8,GROUP=40,RINGS=8',f'CONSUMERS={self.consumers},GROUP={32+self.consumers},RINGS={self.slots}')
  src=src.replace('TMN_DEVI void ring_source', (R/'packed_glu.cuh').read_text()+'''
TMN_DEVI void ring_store(const Params& p,uint8_t* sm,int cid,int slot,int rank){
 for(int i=0;i<2;++i){int plane=(cid*RINGS+slot)*4+(rank/16)*2+i;
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%2,%3,%4}],[%1];"::"l"(&p.ringstore),"r"(smem_u32(sm+(1-i)*4096)),"r"(0),"r"((rank%16)*32),"r"(plane):"memory");
 }tma_store_commit();
}
TMN_DEVI void ring_source''')
  src=re.sub(r'  // Shared derivative matrix.*?  fence_regs\(dw0\);', '''  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_glu(pre,xn,sm+DERIV,ma,mb);__syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0)ring_store(p,sm+DERIV,cid,it%RINGS,rank);
  fence_regs(dw0);''',src,flags=re.S)
  src=src.replace('wgmma_commit();wgmma_wait<0>();fence_regs(dw0);','wgmma_commit();if(tid==0){asm volatile("cp.async.bulk.wait_group 0;":::"memory");signal_flag(flags+rank,it+1);}wgmma_wait<0>();fence_regs(dw0);')
  src=src.replace('p.cohorts',str(self.cohorts))
  body=(PRE/'ring.cu').read_text().replace('#include "ring_source.cuh"',src).replace('p.cohorts',str(self.cohorts)).replace('mw_d256_b7_ring','mw_d256_b7_ring2')
  out=T.compile_text(body,flags);self.k=T.load_unit(str(out),'mw_d256_b7_ring2').kernel('mw_d256_b7_ring2');self.k.set_max_dynamic_smem(114816)
  body=(PRE/'ring_reduce.cu').read_text().replace('#include "base.cuh"',(PRE/'base.cuh').read_text())
  out=T.compile_text(body,flags);self.reduce=T.load_unit(str(out),'mw_d256_b7_reduce').kernel('mw_d256_b7_reduce');self.reduce.set_max_dynamic_smem(W.tuning(256)[2])
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),128,114816)))
  assert self.ctas<=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ,(self.ctas,occ)
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  rm=lambda channels:L.tensor_map(self.ring,[64,channels,1],dims=[64,512,self.cohorts*self.slots*4],strides_bytes=[128,65536],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),self.mask,*p.gp,p.floats[7],p.M,rm(64),rm(32),p.maps[5],p.maps[2],*p.maps[10:14],p.tensors[10],self.ring,self.flags,self.cohorts])
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.ctas,1,1,128,1,1,114816,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
  W.launch(self.reduce,self.p.params7,264,D=256);return self.p.outputs
