"""Sequential bounded GP buffer: native source, ordered Lt dX, FP32 dW carry."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from wide_mask_transform import mask_stage
from lt_contract import LtBmm
from chunk_input_ln import InputOnly

class ChunkL2:
    def __init__(self,plan,rows=8192,splits=4):
        self.plan=plan;self.p=p=plan.p;self.rows=rows;self.splits=splits
        assert p.D==256 and p.M%rows==0 and rows%(64*splits)==0
        root=Path(__file__).resolve().parent;d=p.D;L=T._launch_module()
        self.input=p.x.new_empty((9*d,rows));self.gp=self.input[d:].view(4,2*d,rows)
        self.partial=torch.empty((splits,8*d,d),device=p.x.device,dtype=torch.float32)
        self.carry=torch.empty((8*d,d),device=p.x.device,dtype=torch.float32)
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        body=mask_stage(plan.b7.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('int M;};','int M,start,count;};')
        body=body.replace('tiles=p.M/64','tiles=p.count/64')
        body=body.replace('begin=(tiles*split)/WEIGHT_SPLITS,end=(tiles*(split+1))/WEIGHT_SPLITS',
                          'begin=p.start/64+(tiles*split)/WEIGHT_SPLITS,end=p.start/64+(tiles*(split+1))/WEIGHT_SPLITS')
        body=body.replace('begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS',
                          'begin=p.start/64+tiles*split/WEIGHT_SPLITS,end=p.start/64+tiles*(split+1)/WEIGHT_SPLITS')
        body=body.replace('sm+DERIV+4096,row,(rank%16)*32','sm+DERIV+4096,row-p.start,(rank%16)*32')
        body=body.replace('sm+DERIV,row,(rank%16)*32','sm+DERIV,row-p.start,(rank%16)*32')
        body=body.replace('size_t(split)*11*D*D+(3+2*which)*D*D','size_t(split)*8*D*D+2*which*D*D')
        body=body.replace('mw_d256_b7_tma','mw_d256_chunk_source')
        body=body.replace('cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];',
            '{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],pol;}')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}']
        self.source_cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.source_cubin),'mw_d256_chunk_source').kernel('mw_d256_chunk_source');self.source.set_max_dynamic_smem(115072)
        cp=(root/'prefix_transpose.cu').read_text().replace('int M,D;','int M,D,start;').replace('bar,col,row);','bar,col,row+p.start);').replace('mw_prefix_transpose','mw_chunk_prefix')
        self.copy_cubin=T.compile_text(cp,flags);self.copy=T.load_unit(str(self.copy_cubin),'mw_chunk_prefix').kernel('mw_chunk_prefix');self.copy.set_max_dynamic_smem(8320)
        reduce='''
#include "tmn_kernels.cuh"
using namespace tmn;using bf=__nv_bfloat16;
struct R{const float* part;float* carry;bf* out[4];int first,last;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_chunk_dw_reduce(__grid_constant__ const R p){
 for(int i=blockIdx.x*256+threadIdx.x;i<524288;i+=gridDim.x*256){
  float v=p.first?0:p.carry[i];
  #pragma unroll
  for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.part[s*524288+i];
  if(p.last)p.out[i/131072][i%131072]=__float2bfloat16_rn(v);
  else p.carry[i]=v;
 }
}
'''
        self.reduce_cubin=T.compile_text(reduce,flags);self.reduce=T.load_unit(str(self.reduce_cubin),'mw_chunk_dw_reduce').kernel('mw_chunk_dw_reduce')
        self.ln=InputOnly(p,16,128,4,splits=8)
        gm=[L.tensor_map(t,[64,32],dims=[rows,512],strides_bytes=[rows*2],swizzle='128B',l2='128B') for t in self.gp]
        self.source_params=[];self.copy_params=[];self.reduce_params=[];self.ops=[]
        for start in range(0,p.M,rows):
            fields=plan.b7.params.fields.copy();fields[4:8]=gm;fields[9:13]=list(self.gp);fields[13]=self.partial
            fields.extend((start,rows));self.source_params.append(L.Struct(fields))
            self.copy_params.append(L.Struct([W.tm(p.dg),W.tm(self.input[:d]),rows,d,start]))
            self.reduce_params.append(L.Struct([self.partial,self.carry,*p.tensors[17:21],int(start==0),int(start+rows==p.M)]))
            self.ops.append(LtBmm(self.input.t().unsqueeze(0),plan.dx.weights.unsqueeze(0),p.tensors[10][start:start+rows].unsqueeze(0),self.workspace))
        self.copy_grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*4

    def stage(self,i):
        self.source.launch((32*self.splits,1,1),(256,1,1),[self.source_params[i]],115072)
        self.copy.launch((self.copy_grid,1,1),(128,1,1),[self.copy_params[i]],8320)

    def select(self,index):
        algo=list(self.ops[0].heuristics[index].algo.data)
        for op in self.ops:
            op.index=index;assert list(op.heuristics[index].algo.data)==algo

    def __call__(self):
        self.plan.dx.pack_weights()
        for i,op in enumerate(self.ops):
            self.stage(i);op();self.reduce.launch((264,1,1),(256,1,1),[self.reduce_params[i]],0)
        self.ln()
