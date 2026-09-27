"""Reuse one compact GP/dX input buffer across complete spatial chunks."""
import torch
from d256_cache_chunked_lt import CacheChunkedLt
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage
from lt_contract import LtBmm


class ReusedChunkLt(CacheChunkedLt):
    def __init__(self,plan,splits=128,chunks=16,evict_last=False):
        super().__init__(plan,splits,chunks,False)
        p=plan.p;prior=plan.b7;rows=p.M//chunks
        assert p.M%chunks==0 and rows%64==0
        self.rows=rows;self.input=torch.empty((9*p.D,rows),device=p.x.device,dtype=torch.bfloat16)
        self.gp=list(self.input[p.D:].view(4,2*p.D,rows).unbind(0))
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        assert body.count('int M;};')==1 and body.count('split=blockIdx.x/32')==2
        body=body.replace('int M;};','int M;int split_offset;int row_begin;};')
        body=body.replace('split=blockIdx.x/32','split=blockIdx.x/32+p.split_offset')
        for point in ('sm+DERIV+4096,row,','sm+DERIV,row,'):
            assert body.count(point)==1
            body=body.replace(point,point.replace(',row,',',row-p.row_begin,'))
        if evict_last and '.bulk_group.L2::cache_hint' not in body:
            instruction='cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];'
            assert body.count(instruction)==1
            body=body.replace(instruction,'{.reg .b64 pol;createpolicy.fractional.L2::evict_last.b64 pol,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],pol;}')
        body=body.replace('mw_d256_b7_tma','mw_d256_reused_chunk_source')
        body+='''
extern "C" __global__ void mw_copy_compact_prefix(const uint4* src,uint4* dst,int begin,int M,int rows){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<256*(rows/8);i+=gridDim.x*blockDim.x){
  int c=i/(rows/8),r=i%(rows/8);dst[i]=src[c*(M/8)+begin/8+r];
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_reused_chunk_source')
        self.k=unit.kernel('mw_d256_reused_chunk_source');self.k.set_max_dynamic_smem(self.smem)
        self.prefix=unit.kernel('mw_copy_compact_prefix')
        for old in self.dx:old.close()
        self.dx=[];self.params=[]
        L=T._launch_module();tm=lambda t:L.tensor_map(t,[64,32],dims=[rows,512],strides_bytes=[rows*2],swizzle='128B',l2='128B')
        for i,(begin,end) in enumerate(self.ranges):
            assert end-begin==rows
            fields=list(prior.params.fields);fields[4:8]=[tm(t) for t in self.gp];fields[9:13]=self.gp;fields[-2]=self.partial
            self.params.append(L.Struct([*fields,i*(splits//chunks),begin]))
            self.dx.append(LtBmm(self.input.t().unsqueeze(0),plan.dx.weights.unsqueeze(0),p.tensors[10][begin:end].unsqueeze(0),self.workspace))
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def set_algorithm(self,data):
        for op in self.dx:
            for index in op.indices:
                if list(op.heuristics[index].algo.data)==data:
                    op.index=index;break
            else:
                op.index=0
                for j,value in enumerate(data):op.heuristics[0].algo.data[j]=value

    def __call__(self):
        self.pack_weights();p=self.plan.p
        for i,(begin,end) in enumerate(self.ranges):
            self.prefix.launch((132,1,1),(256,1,1),[self.plan.dx.input,self.input,begin,p.M,self.rows],0)
            self.k.launch((32*(self.splits//self.chunks),1,1),(256,1,1),[self.params[i]],self.smem)
            self.dx[i]()
