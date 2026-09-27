"""Duplicate projection for half-column dW CTAs and fit three resident CTAs."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class HalfColumnsSource:
    def __init__(self,plan,packed=True):
        prior=plan.b7;self.__dict__.update(prior.__dict__)
        body=mask_stage(prior.original.source_text,'bulk')
        body=body.replace('sm+BAR+128+slot*128','sm+73856')
        body=body.replace('INPUT=36864,WEIGHT=73728,DERIV=106496','INPUT=40960,WEIGHT=40960,DERIV=32768')
        body=body.replace('sm+114688','sm+73728')
        body=body.replace('mbar_arrive_expect_tx(bar+slot,INPUT+128);','mbar_arrive_expect_tx(bar+slot,36864+128);')
        body=body.replace('split=blockIdx.x/32','split=blockIdx.x/64')
        marker=' int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=blockIdx.x%32,split=blockIdx.x/64;'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n int column=(blockIdx.x/32)%2;')
        body=body.replace('int slot=it%2','int slot=0')
        body=body.replace('mbar_wait(bar+slot,(it/2)&1)','mbar_wait(bar+slot,it&1)')
        body=body.replace('if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1)','if(it>=1)mbar_wait(bar+3,(it-1)&1)')
        # Read both incoming dL/dR fragments before aliasing them with dGate.
        a=' #pragma unroll\n for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+32768)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));'
        assert body.count(a)==1
        b=''' uint32_t incoming[2][4];
 #pragma unroll
 for(int q=0;q<2;++q)ldsm_x4_t(incoming[q],smem_u32(s+32768)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
 named_bar_sync(1,128);
 #pragma unroll
 for(int q=0;q<2;++q){auto& dy=incoming[q];uint32_t dg[4],dp[4];'''
        body=body.replace(a,b)
        if packed:
            body=body.replace('void packed_glu(float (&a)[32]','void packed_glu(uint32_t (&a)[16]')
            body=body.replace('pack_bf16(a[q*8+j*2],a[q*8+j*2+1])','a[q*4+j]')
            body=body.replace('pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1])','a[(q+2)*4+j]')
            marker='});wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
            assert body.count(marker)==1
            body=body.replace(marker,marker+'\n  uint32_t prepack[16];\n  #pragma unroll\n  for(int q=0;q<16;++q)prepack[q]=pack_bf16(pre[2*q],pre[2*q+1]);')
            body=body.replace('packed_glu(pre,xn','packed_glu(prepack,xn')
        body=body.replace('float dw0[64]={},dw1[64]={};','float dw0[32]={},dw1[32]={};')
        body=body.replace('mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),','mma64_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn+column*16384),')
        body=body.replace('mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),','mma64_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+column*16384+8192),')
        body=body.replace('if(tid==0){store_gp','if(tid==0&&column==0){store_gp')
        body=body.replace(' for(int j=0;j<64;++j){',' static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;')
        body=body.replace('  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];\n }','  p.part[ix+column*128]=dw0[j];p.part[ix+column*128+64]=dw1[j];\n });')
        body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(256,3)').replace('setmaxnreg_inc<224>()','setmaxnreg_inc<128>()')
        body=body.replace('mw_d256_b7_tma','mw_d256_half_columns_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_half_columns_source').kernel('mw_d256_half_columns_source');self.smem=73984;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
    def __call__(self):self.k.launch((64*self.splits,1,1),(256,1,1),[self.params],self.smem)
