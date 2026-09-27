"""Two streamed N128 groups target two CTAs while retaining all dNorm in shared."""
from wide_full_width_dn_ln import FullWidthDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class HalfWidthDnLN(FullWidthDnLN):
    def __init__(self,plan,emit_dn=False):
        super().__init__(plan,emit_dn,32);assert plan.p.D==256
        body=self.source_text.replace('NT=512','NT=256').replace('__launch_bounds__(NT,1)','__launch_bounds__(NT,2)')
        body=body.replace('INPUT=4096+64*H','INPUT=4096+64*D').replace('for(int col=0;col<H;col+=H)','for(int col=0;col<H;col+=D)')
        marker='for(int c=0;c<H;c+=64)tma_load_2d(sm+slot*INPUT+4096+'
        assert body.count(marker)==1;body=body.replace(marker,marker.replace('c<H','c<D'))
        body=body.replace('mw_wide_full_width_dn_ln','mw_d256_half_width_dn_ln');self.source_text=body
        self.threads=256;self.smem=8192+384*256+128+32*8
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_half_width_dn_ln')
        self.k=unit.kernel('mw_d256_half_width_dn_ln');self.k.set_max_dynamic_smem(self.smem)
        self.reduce_first=unit.kernel('mw_wide_stream_ln_reduce_first');self.reduce_last=unit.kernel('mw_wide_stream_ln_reduce_last')
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
