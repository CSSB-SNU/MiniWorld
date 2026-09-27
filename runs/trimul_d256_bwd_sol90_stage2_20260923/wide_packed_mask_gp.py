"""Lossless two-bit mask codes; nonbinary elements use the original BF16 mask."""
from pathlib import Path
import torch
from wide_staged_epilogue_gp import StagedEpilogueGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PackedMaskGP:
    def __init__(self,plan):
        prior=StagedEpilogueGP(plan);self.__dict__.update(prior.__dict__)
        self.mask=plan.f.mask;self.packed=torch.empty((self.p.n,self.p.n//16),device=self.mask.device,dtype=torch.int32)
        body=prior.source_text
        marker='outmap[4];int N;';assert body.count(marker)==1
        body=body.replace(marker,'outmap[4];CUtensorMap packedmask;int N;')
        marker='mbar_arrive_expect_tx(bar+6,98304);';assert body.count(marker)==1
        body=body.replace(marker,'mbar_arrive_expect_tx(bar+6,65536);')
        marker=' if(threadIdx.x==0)load_epi<MODE,2>(p,sm,bar,ch,mi,ni);';assert body.count(marker)==1
        body=body.replace(marker,'')
        marker=' mbar_wait(bar+6,0);__syncthreads();';assert body.count(marker)==1
        body=body.replace(marker,' mbar_wait(bar+6,0);mbar_wait(bar+7,0);__syncthreads();')
        marker='if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);}'
        assert body.count(marker)==1
        body=body.replace(marker,'''if(threadIdx.x==0){
  mbar_arrive_expect_tx(bar+7,4096);
  tma_load_2d(sm+98432,&p.packedmask,bar+7,ni/16,mi);
  load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);
 }''')
        marker='for(int i=0;i<7;++i)mbar_init';assert body.count(marker)==1
        body=body.replace(marker,'for(int i=0;i<8;++i)mbar_init')
        marker='uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+65536+off);'
        assert body.count(marker)==1
        body=body.replace(marker,'''uint32_t raw=pack_bf16(v[j],v[j+1]),masked;
  uint32_t codes=(reinterpret_cast<const uint32_t*>(sm+98432)[(WG*64+r)*8+c/16]>>(2*(c%16)))&15u;
  uint32_t mask=(codes&1u)*0x3f80u+((codes>>2)&1u)*0x3f800000u;
  if(codes&10u)mask=*reinterpret_cast<const uint32_t*>(p.mask+size_t(mi+WG*64+r)*p.N+ni+c);''')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_packed_mask_gp')
        body+='''
__device__ __forceinline__ uint32_t spread16(uint32_t v){
 v=(v|(v<<8))&0x00ff00ffu;v=(v|(v<<4))&0x0f0f0f0fu;
 v=(v|(v<<2))&0x33333333u;return (v|(v<<1))&0x55555555u;
}
extern "C" __global__ void mw_pack_gp_mask(const uint16_t* src,uint32_t* dst,int count){
 int lane=threadIdx.x%32;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){
  uint32_t raw=src[i];uint32_t lo=__ballot_sync(0xffffffffu,raw!=0),hi=__ballot_sync(0xffffffffu,raw!=0&&raw!=0x3f80u);
  if(lane==0||lane==16)dst[i/16]=spread16((lo>>lane)&65535u)|(spread16((hi>>lane)&65535u)<<1);
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_packed_mask_gp')
        self.k=unit.kernel('mw_wide_packed_mask_gp');self.pack=unit.kernel('mw_pack_gp_mask')
        self.smem=98432+4096;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();tm=L.tensor_map(self.packed,[8,128],dims=[self.p.n//16,self.p.n],strides_bytes=[self.p.n//4],swizzle='none',l2='128B')
        fields=prior.params.fields.copy();fields.insert(len(fields)-1,tm);self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def encode(self):self.pack.launch((132,1,1),(256,1,1),[self.mask,self.packed,T._launch_module().i32(self.mask.numel())],0)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
