"""Reuse D256's immediate-offset WGMMA descriptors at wider dimensions."""
from pathlib import Path
import os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def offsets(body):
    root=Path(__file__).resolve().parent
    body=body.replace('// MMA_HELPERS','// MMA_HELPERS\n'+(root/'mma_offset.cuh').read_text())
    # Helpers may have already been expanded by the pipelined builder.
    if 'void mma64_off' not in body:
        point='constexpr int D=WIDTH'
        assert point in body
        body=body.replace(point,(root/'mma_offset.cuh').read_text()+'\n'+point)
    old='mma64<0,0>(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+(k/4)*8192+(k%4)*32),16,1024,1),k>0);'
    new='mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);'
    assert body.count(old)==1
    body=body.replace(old,new)
    old='mma64<0,1>(dw[c],smem_desc(smem_u32(sm+DERIV+k*32),16,1024,1),\n      smem_desc(smem_u32(xn+(wg*NC+c)*8192+k*2048),8192,1024,1),it>0||k>0);'
    if old in body:
        body=body.replace(old,'mma64_off<k*32,c*8192+k*2048,0,1>(dw[c],smem_desc(smem_u32(sm+DERIV),16,1024,1),smem_desc(smem_u32(xn+wg*NC*8192),8192,1024,1),it>0||k>0);')
    else:
        old='mma64<0,1>(dw[c],smem_desc(smem_u32(dg+k*32),16,1024,1),smem_desc(smem_u32(xn+(WG*NC+c)*8192+k*2048),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1
        body=body.replace(old,'mma64_off<k*32,c*8192+k*2048,0,1>(dw[c],smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*NC*8192),8192,1024,1),it>0||k>0);')
    return body


class OffsetSource:
    def __init__(self,p,original):
        self.p=p;self.original=original
        root=Path(__file__).resolve().parent;pre=root.parent/'trimul_d256_bwd_sol90_20260923'
        body=offsets((root/'wide_source.cu').read_text())
        body=body.replace('// MMA_HELPERS',(pre/'mma.cuh').read_text())
        body=body.replace('// PACKED_GLU',(root/'packed_glu.cuh').read_text().replace('s+32768','s+CH'))
        if os.environ.get('WIDE_SOURCE_N128')=='1':
            from wide_source_mma128 import widen_dw
            body=widen_dw(body)
        body=body.replace('mw_wide_b7_source','mw_wide_offset_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DWEIGHT_SPLITS=32']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_offset_source').kernel('mw_wide_offset_source')
        self.smem=original.source_smem;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):
        self.k.launch((self.p.D//8*32,1,1),(256,1,1),[self.original.params],self.smem)
