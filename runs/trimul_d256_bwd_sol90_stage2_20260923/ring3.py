"""Bounded L2 ring with TMA producers in both source and dX/LN CTAs."""
from pathlib import Path
import os,re,ctypes,torch
from tma_b7 import TmaB7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class Ring3:
 def __init__(self,p,leaves):
  assert os.environ.get('GP_WARP')=='1' and os.environ.get('GP_OFF')=='1'
  self.p=p;self.cohorts=int(os.environ.get('RCOHORTS','4'));self.consumers=int(os.environ.get('RCONSUMERS','32'));self.slots=int(os.environ.get('RSLOTS','32'));self.windows=int(os.environ.get('RWINDOWS','1' if p.n==384 else '4'))
  self.splits=self.cohorts*self.windows;assert self.splits<=32
  native=TmaB7(p,leaves,packed=True);self.mask=native.mask
  src=native.source_text.rsplit('extern "C"',1)[0]
  src=src.replace('int M;};','int M;CUtensorMap ringmap,ringstore,map[16],xmap,resmap,dxmap;bf* t[24];float* f[13];int L,startTile,windowTiles,partBase;unsigned* flags;};')
  extras=f'constexpr int COHORTS={self.cohorts},CONSUMERS={self.consumers},GROUP={32+self.consumers},RSLOTS={self.slots},STAGE=40960;\n'
  extras+='''
TMN_DEVI void wait_flag(unsigned* ptr,unsigned want){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got<want)__nanosleep(32);}while(got<want);}
TMN_DEVI void signal_flag(unsigned* ptr,unsigned val){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(ptr),"r"(val):"memory");}
TMN_DEVI void ring_store(const Params& p,uint8_t* sm,int cid,int slot,int rank){
 for(int i=0;i<2;++i){int plane=(cid*RSLOTS+slot)*4+(rank/16)*2+i;
  asm volatile("{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3,%4}],[%1],pol;}"::"l"(&p.ringstore),"r"(smem_u32(sm+(1-i)*4096)),"r"(0),"r"((rank%16)*32),"r"(plane):"memory");
 }tma_store_commit();
}
'''
  src=src.replace('TMN_DEVI void compute_source',extras+'\nTMN_DEVI void compute_source')
  src=src.replace('rank=blockIdx.x%32,split=blockIdx.x/32','rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP,split=p.partBase+cid')
  src=src.replace('int tiles=p.M/64,begin=(tiles*split)/WEIGHT_SPLITS,end=(tiles*(split+1))/WEIGHT_SPLITS;','int begin=0,end=p.windowTiles/COHORTS;')
  src=src.replace('row=tile*64;','row=(p.startTile+cid+tile*COHORTS)*64;')
  src=src.replace('if(tid==0){store_gp(p.gmap+(rank/16)*2,sm+DERIV+4096,row,(rank%16)*32);store_gp(p.gmap+(rank/16)*2+1,sm+DERIV,row,(rank%16)*32);tma_store_commit();}','if(tid==0){if(it>=RSLOTS)wait_flag(p.flags+(cid*RSLOTS+it%RSLOTS)*33+32,it-RSLOTS+1);ring_store(p,sm+DERIV,cid,it%RSLOTS,rank);}')
  src=src.replace('wgmma_commit();wgmma_wait<0>();fence_regs(dw0);','wgmma_commit();if(tid==0){asm volatile("cp.async.bulk.wait_group 0;":::"memory");signal_flag(p.flags+(cid*RSLOTS+it%RSLOTS)*33+rank,it+1);}wgmma_wait<0>();fence_regs(dw0);')
  src+='''
TMN_DEVI void producer_source(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x)return;int rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP;
 mbar_arrive_expect_tx(bar+2,CH);for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
 for(int it=0;it<p.windowTiles/COHORTS;++it){int slot=it%2;if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);load_input(p,sm,bar,(p.startTile+cid+it*COHORTS)*64,rank,slot);}
}
'''
  dx=(R/'dx_ln_warp.cu').read_text()
  helpers=dx[dx.index('TMN_DEVI float rd'):dx.index('TMN_DEVI void producer')]
  dx=dx[dx.index('TMN_DEVI void producer'):dx.index('extern "C" __global__')]
  dx=dx.replace('if(threadIdx.x)return;','if(threadIdx.x>=32)return;int lane=threadIdx.x;')
  loop='for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){'
  dx=dx.replace(loop,'int cid=blockIdx.x/GROUP,crank=blockIdx.x%GROUP-32;\n for(int seq=crank,round=0;seq<p.windowTiles/COHORTS;seq+=CONSUMERS,++round){int row=(p.startTile+cid+seq*COHORTS)*64;')
  dx=dx.replace('if(round)mbar_wait(bars+5,(round-1)&1);','if(round&&lane==0)mbar_wait(bars+5,(round-1)&1);\n  __syncwarp();wait_flag(p.flags+(cid*RSLOTS+seq%RSLOTS)*33+lane,seq+1);__syncwarp();')
  dx=dx.replace('for(int step=0;step<36;++step){int slot=step%2,it=round*36+step;','if(lane==0)for(int step=0;step<36;++step){int slot=step%2,it=round*36+step;')
  dx=dx.replace('else tma_load_2d(dst,&p.map[6+plane],bars+slot,row,k);','else tma_load_3d(dst,&p.ringmap,bars+slot,0,k,(cid*RSLOTS+seq%RSLOTS)*4+plane);')
  dx=dx.replace('  static_for<64>([&](auto jj){','  if(tid==0)signal_flag(p.flags+(cid*RSLOTS+seq%RSLOTS)*33+32,seq+1);\n  static_for<64>([&](auto jj){')
  dx=dx.split(' for(int which=0;which<4;++which)')[0]+'}\n'
  src+=helpers+'\n'+dx+'''
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_ring3(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+114688);int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<6;++i)mbar_init(bars+i,1);fence_barrier_init();}
 for(int i=blockIdx.x*256+tid;i<COHORTS*RSLOTS*33;i+=gridDim.x*256)p.flags[i]=0;
 if(p.partBase==0)for(int c=blockIdx.x*256+tid;c<D;c+=gridDim.x*256){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_dec<32>();if(blockIdx.x%GROUP<32)producer_source(p,sm,bars);else producer(p,sm,bars);}
 else{setmaxnreg_inc<224>();if(blockIdx.x%GROUP<32)compute_source(p);else consumer(p,sm,bars);}
}
'''
  src='#include <cooperative_groups.h>\n'+src
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}','-DWIDTH=256']
  out=T.compile_text(src,flags);self.k=T.load_unit(str(out),'mw_d256_ring3').kernel('mw_d256_ring3');self.k.set_max_dynamic_smem(114816)
  self.ring=p.x.new_empty((self.cohorts,self.slots,4,512,64));self.flags=torch.zeros((self.cohorts,self.slots,33),device=p.x.device,dtype=torch.int32);self.ctas=self.cohorts*(32+self.consumers)
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,114816)));assert self.ctas<=132*occ,(self.ctas,occ)
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B');rm=lambda c:L.tensor_map(self.ring,[64,c,1],dims=[64,512,self.cohorts*self.slots*4],strides_bytes=[128,65536],swizzle='128B',l2='128B')
  t7=p.tensors.copy();t7[22:24]=[p.dl,p.dr];self.params=[]
  for window in range(self.windows):
   self.params.append(L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),*[dy(x) for x in p.gp],self.mask,*p.gp,p.floats[7],p.M,rm(64),rm(32),*p.maps7,W.tm(p.x.reshape(p.M,256)),W.tm(p.dy.reshape(p.M,256)),W.tm(p.dx.reshape(p.M,256)),*t7,*p.floats,p.n,window*(p.M//64//self.windows),p.M//64//self.windows,window*self.cohorts,self.flags]))
  red=(PRE/'base.cuh').read_text()+'''\nextern "C" __global__ void mw_d256_ring3_reduce(__grid_constant__ const Params p){for(int i=0;i<4;++i)reduce_w(p,(H+D+i*H)*D,H*D,p.t[17+i]);}\n'''
  out=T.compile_text(red,flags);self.reduce=T.load_unit(str(out),'mw_d256_ring3_reduce').kernel('mw_d256_ring3_reduce')
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv
  for p in self.params:
   args=L._Packed([p]);drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.ctas,1,1,256,1,1,114816,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
  self.reduce.launch((264,1,1),(256,1,1),[self.p.params7],0);return self.p.outputs
