"""N128 channel CTAs with DSM LN sums; no dXn HBM intermediate."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent

class DxClusterLN:
 def __init__(self,p,splits,slots=2):
  assert slots in (2,3) and hasattr(p,'input_stats')
  body=(R/'dx_separate.cu').read_text().split('#if DX_SEP_WS\nextern "C"')[0]
  body=body.replace('#include "mma.cuh"',(R.parent/'trimul_d256_bwd_sol90_20260923'/'mma.cuh').read_text())
  body=body.replace('CUtensorMap dxn;','CUtensorMap dxn,xmap,resmap;')
  body=body.replace('TMN_DEVI void consume(', (R/'dx_cluster_ln.cuh').read_text()+'\nTMN_DEVI void consume(')
  hook=' int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;'
  assert body.count(hook)==1
  body=body.replace(hook,hook+'float gg[4]={},bb[4]={};')
  old='''  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<128;c+=64)store_tile(&p.dxn,sm+c*128,col+c,row);tma_store_commit();tma_store_wait_all();}named_bar_sync(1,128);
  if(tid==0)mbar_arrive(bars+2*NSLOT);'''
  assert old in body
  body=body.replace(old,'  __syncthreads();cluster_ln(p,sm,bars,row,col,round,gg,bb);')
  # Affine sums stay private to each channel half. Final reduction reads only
  # its own CTA. Last cluster barrier guarantees peer accesses have ended.
  end=body.rfind('\n}')
  body=body[:end]+'''
 float* part=reinterpret_cast<float*>(sm);int col=(blockIdx.x%2)*128;
 for(int q=0;q<4;++q){int c=lane+q*32;part[warp*128+c]=gg[q];part[(4+warp)*128+c]=bb[q];}__syncthreads();
 float g=0,b=0;for(int w=0;w<4;++w){g+=part[w*128+tid];b+=part[(4+w)*128+tid];}atomicAdd(p.f[8]+col+tid,g);atomicAdd(p.f[9]+col+tid,b);
 for(int which=0;which<4;++which){int offset=(H+D+which*H)*D;
  for(int i=blockIdx.x*128+tid;i<H*D;i+=gridDim.x*128){float v=0;for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];p.t[17+which][i]=cv(v);}
 }
''' +body[end:]
  body+='''
extern "C" __global__ void mw_d256_dx_cluster_zero(__grid_constant__ const Params p){int c=threadIdx.x;p.f[8][c]=0;p.f[9][c]=0;}
extern "C" __global__ __launch_bounds__(128,4) void mw_d256_dx_cluster_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+NSLOT*STAGE);
 if(threadIdx.x==0){for(int i=0;i<2*NSLOT+1;++i)mbar_init(bars+i,1);fence_barrier_init();}
 reinterpret_cast<float*>(sm+NSLOT*STAGE+640)[threadIdx.x]=p.f[0][(blockIdx.x%2)*128+threadIdx.x];
 __syncthreads();cooperative_groups::this_cluster().sync();consume(p,sm,bars);cooperative_groups::this_cluster().sync();
}
'''
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}',f'-DDX_SEP_SLOTS={slots}','-DDX_SEP_WS=0']
  self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_dx_cluster_ln')
  self.k=unit.kernel('mw_d256_dx_cluster_ln');self.zero=unit.kernel('mw_d256_dx_cluster_zero');self.smem=slots*24576+2048;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();ff=p.floats.copy();ff[12]=p.input_stats
  self.params=L.Struct([*p.maps7,*p.tensors,*ff,p.M,p.n,W.tm(p.dx.reshape(p.M,256)),W.tm(p.x.reshape(p.M,256)),W.tm(p.dy.reshape(p.M,256))])
  self.args=L._Packed([self.params]);self.drv=unit.drv;d=self.drv.d
  attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
  attr.value.clusterDim.x=2;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
  cfg=d.CUlaunchConfig();cfg.gridDimX=528;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=128;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
  self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(d.CUfunction(int(self.k.handle)),cfg)))
  assert self.active_clusters>0
  cfg.gridDimX=2*self.active_clusters;self.grid=cfg.gridDimX;self.cfg=cfg;self.attr=attr
  self.metadata=dict(cubin=str(self.cubin),smem=self.smem,grid=self.grid,active_clusters=self.active_clusters,slots=slots)
 def __call__(self):
  self.zero.launch((1,1,1),(256,1,1),[self.params],0)
  self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
  self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
