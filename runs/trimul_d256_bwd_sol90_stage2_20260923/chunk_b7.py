"""Sequential L2-sized producer/consumer windows without a spinning ring."""
from pathlib import Path
import os,copy,torch
from tma_b7 import TmaB7
from dx_ln import DxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent
class ChunkB7:
 def __init__(self,p,leaves):
  assert os.environ.get('DX_WARP')=='1' and os.environ.get('DX_NO_DW')=='1'
  self.p=p;self.width=int(os.environ.get('CHUNK_ROWS','8192'));self.splits=int(os.environ.get('CHUNK_SPLITS','8'));assert p.M%self.width==0
  self.count=p.M//self.width;self.gp=p.x.new_empty((4,512,self.width));self.part=torch.empty((self.count*self.splits,11*256*256),device=p.x.device,dtype=torch.float32)
  self.gamma=torch.empty((self.count,256),device=p.x.device,dtype=torch.float32);self.beta=torch.empty_like(self.gamma);self.steps=[]
  for i in range(self.count):
   lo=i*self.width;hi=lo+self.width;q=copy.copy(p);q.M=self.width
   window=lambda t:t.reshape(p.M,256)[lo:hi]
   q.x=window(p.x);q.dy=window(p.dy);q.xn=window(p.xn);q.dx=window(p.dx);q.dg=window(p.dg);q.dl=p.dl.reshape(512,p.M)[:,lo:hi];q.dr=p.dr.reshape(512,p.M)[:,lo:hi];q.mask=p.mask.reshape(p.M)[lo:hi]
   q.gp_all=self.gp;q.gp=list(self.gp.unbind());q.floats=p.floats.copy();q.floats[7]=self.part[i*self.splits:(i+1)*self.splits];q.floats[8]=self.gamma[i];q.floats[9]=self.beta[i]
   q.tensors=p.tensors.copy();q.tensors[0]=q.x;q.tensors[2]=q.dy;q.tensors[5]=q.xn;q.tensors[8]=q.dg;q.tensors[11]=q.dx;q.tensors[13:17]=q.gp
   q.maps=p.maps.copy();q.maps[0]=W.tm(q.xn);q.maps[5]=W.tm(q.dg);q.maps[6:10]=[W.tm(g) for g in q.gp];q.maps7=q.maps.copy()
   source=TmaB7(q,leaves,packed=True,splits=self.splits);source.wide_finish=DxLN(q,self.splits);self.steps.append(source)
  body='''#include <cuda_bf16.h>
struct Params{float* part;float* gamma;float* beta;__nv_bfloat16* dw[4];float* dg;float* db;int count,splits;};
extern "C" __global__ void mw_d256_chunk_reduce(__grid_constant__ const Params p){
 int tid=blockIdx.x*256+threadIdx.x;
 for(int i=tid;i<4*512*256;i+=gridDim.x*256){int which=i/(512*256),u=i%(512*256);float sum=0;for(int s=0;s<p.count*p.splits;++s)sum+=p.part[size_t(s)*11*256*256+(3+2*which)*256*256+u];p.dw[which][u]=__float2bfloat16_rn(sum);}
 if(tid<256){float g=0,b=0;for(int i=0;i<p.count;++i){g+=p.gamma[i*256+tid];b+=p.beta[i*256+tid];}p.dg[tid]=g;p.db[tid]=b;}
}
'''
  out=T.compile_text(body,['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v']);self.reduce=T.load_unit(str(out),'mw_d256_chunk_reduce').kernel('mw_d256_chunk_reduce')
  self.params=T._launch_module().Struct([self.part,self.gamma,self.beta,*p.dw,p.floats[8],p.floats[9],self.count,self.splits])
 def __call__(self):
  for source in self.steps:source()
  self.reduce.launch((264,1,1),(256,1,1),[self.params],0);return self.p.outputs
