"""Prefetch the byte mask and gate plane before the final MMA retires."""
from wide_byte_mask_gp import ByteMaskGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class StagedByteMaskGP(ByteMaskGP):
    def __init__(self,plan,banded=False):
        super().__init__(plan,banded)
        body=self.source_text
        helper='''
template<int MODE,int PLANE> TMN_DEVI void load_byte_epi(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
  int q=(wm*2+wn)*8192;
  tma_load_2d(sm+PLANE*32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+PLANE*32)*p.N+mi+64*wm);
 }
}
'''
        marker='template<int MODE> TMN_DEVI void consume('
        assert body.count(marker)==1;body=body.replace(marker,helper+marker)
        marker=' if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);}'
        assert body.count(marker)==1
        body=body.replace(marker,''' if(threadIdx.x==0){
  mbar_arrive_expect_tx(bar+6,81920);
  tma_load_2d(sm+65536,&p.maskmap,bar+6,ni,mi);
  load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);
 }''')
        marker='int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n  if(threadIdx.x==0 && ki==p.N-64)load_byte_epi<MODE,0>(p,sm,bar,ch,mi,ni);')
        start=body.index(' // Both consumers finish all WGMMA reads');end=body.index(' mbar_wait(bar+6,0);__syncthreads();',start)
        body=body[:start]+''' __syncthreads();
 if(threadIdx.x==0)load_byte_epi<MODE,1>(p,sm,bar,ch,mi,ni);
'''+body[end:]
        body=body.replace('mw_wide_byte_mask_gp','mw_wide_staged_byte_mask_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_staged_byte_mask_gp').kernel('mw_wide_staged_byte_mask_gp')
        self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
