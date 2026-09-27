"""Move sparse projection escape reads out of the main contraction consumer."""
from pathlib import Path
import torch
from wide_ab_projection_gp import ABProjectionGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ABPatchedProjectionGP(ABProjectionGP):
    def __init__(self,plan):
        super().__init__(plan);root=Path(__file__).resolve().parent;p=plan.p;L=T._launch_module()
        self.capacity=max(65536,4*p.D*p.M//100)
        self.patch_records=torch.empty((self.capacity,2),device=p.x.device,dtype=torch.int32)
        self.patch_count=torch.zeros((),device=p.x.device,dtype=torch.int32)
        body=self.source_text
        body=body.replace('const bf* escapes;int N;','const bf* escapes;const unsigned* patch_count;unsigned capacity;int N;')
        body=body.replace('uint32_t code,const bf* escape){','uint32_t code,const bf* escape,bool sparse){')
        body=body.replace('if(code==3)return __bfloat16_as_ushort(*escape);','if(code==3){if(!sparse)return __bfloat16_as_ushort(*escape);return 0;}')
        marker=' mbar_wait(bar+6,0);__syncthreads();';assert body.count(marker)==1
        body=body.replace(marker,marker+'\n const bool use_patch=(*p.patch_count<=p.capacity);')
        body=body.replace('p.escapes+ai);','p.escapes+ai,use_patch);').replace('p.escapes+ai+1);','p.escapes+ai+1,use_patch);')
        marker='  *reinterpret_cast<uint32_t*>(sm+off)=dp;';assert body.count(marker)==1
        body=body.replace(marker,'''  if(use_patch){
   if((code&3u)==3u)dg=(dg&0xffff0000u)|(masked&65535u);
   if((code>>2)==3u)dg=(dg&65535u)|(masked&0xffff0000u);
  }
'''+marker)
        body=body.replace('mw_wide_ab_projection_gp','mw_wide_ab_patched_projection_gp')
        body+='''
extern "C" __global__ void mw_patch_ab_projection_gp(const uint2* records,const unsigned* count,bf* gp,unsigned capacity){
 unsigned total=*count;if(total>capacity)return;
 for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x){
  uint2 r=records[i];float da=__bfloat162float(gp[r.x]);
  float pr=__uint_as_float(r.y<<16),ga=math::sigmoid(__uint_as_float(r.y&0xffff0000u));
  gp[r.x]=__float2bfloat16_rn(((da*pr)*ga)*(1.f-ga));
 }
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_ab_patched_projection_gp')
        self.k=unit.kernel('mw_wide_ab_patched_projection_gp');self.k.set_max_dynamic_smem(self.smem)
        self.patch=unit.kernel('mw_patch_ab_projection_gp')
        fields=self.params.fields.copy();fields[-1:-1]=[self.patch_count,L.i32(self.capacity)];self.params=L.Struct(fields)
        source=(root/'pack_ab_projection.cu').read_text().replace('#include "ab_projection_codec.cuh"',(root/'ab_projection_codec.cuh').read_text())
        source=source.replace('int M,int D){','int M,int D,uint2* records,unsigned* patch_count,unsigned capacity){')
        marker='   if(code==3)escapes[base+q]=__ushort_as_bfloat16(p);';assert source.count(marker)==1
        source=source.replace(marker,'''   if(code==3){
    escapes[base+q]=__ushort_as_bfloat16(p);
    unsigned dst=atomicAdd(patch_count,1u);
    if(dst<capacity){
     size_t gp_index=size_t(2*(ch/(2*D))+1)*2*D*M+size_t(ch%(2*D))*M+row+q;
     records[dst]=make_uint2(unsigned(gp_index),p|(g<<16));
    }
   }''')
        source=source.replace('mw_pack_ab_projection','mw_pack_ab_patched_projection')
        self.pack_cubin=T.compile_text(source,flags);self.pack=T.load_unit(str(self.pack_cubin),'mw_pack_ab_patched_projection').kernel('mw_pack_ab_patched_projection')
        self.pack_args.extend((self.patch_records,self.patch_count,L.i32(self.capacity)))
        self.patch_args=[self.patch_records,self.patch_count,p.gp_all,L.i32(self.capacity)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def set_capacity(self,capacity):
        assert 0<=capacity<=self.patch_records.shape[0]
        L=T._launch_module();fields=self.params.fields.copy();fields[-2]=L.i32(capacity);self.params=L.Struct(fields)
        self.capacity=capacity;self.pack_args[-1]=L.i32(capacity);self.patch_args[-1]=L.i32(capacity)

    def encode(self):
        self.patch_count.zero_();super().encode()

    def __call__(self):
        super().__call__();self.patch.launch((528,1,1),(256,1,1),self.patch_args,0)
