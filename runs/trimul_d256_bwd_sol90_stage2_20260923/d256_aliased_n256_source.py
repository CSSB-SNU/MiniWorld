"""Two compute warpgroups, aliased dL/dGate storage, two resident CTAs."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class AliasedN256Source:
    def __init__(self,p,original,producer_regs=80):
        self.p=p;self.original=original;root=Path(__file__).resolve().parent
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        helper=(root/'mma256.cuh').read_text()
        body=body.replace('TMN_DEVI void consume',helper+'\nTMN_DEVI void consume')
        body=body.replace('float dw0[64]={},dw1[64]={};','float dw[128]={};').replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
        old='mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1
        body=body.replace(old,'mma256<0,1>(dw,smem_desc(smem_u32(dg+k*32),16,1024,1),smem_desc(smem_u32(xn+k*2048),8192,1024,1),it>0||k>0);')
        old='mma128_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1;body=body.replace(old,'')
        body=body.replace('static_for<64>([&](auto jj)','static_for<128>([&](auto jj)').replace('p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];','p.part[ix]=dw[j];')
        body=body.replace('mw_d256_aliased_pipe_source','mw_d256_aliased_n256_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={original.splits}',f'-DPRODUCER_REGS={producer_regs}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_aliased_n256_source').kernel('mw_d256_aliased_n256_source');self.smem=115072
        self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((32*self.original.splits,1,1),(256,1,1),[self.original.params],self.smem)
