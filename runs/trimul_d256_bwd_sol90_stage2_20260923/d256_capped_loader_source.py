"""One loader warp plus a compute warpgroup, without register redistribution."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage
class CappedLoaderSource:
    def __init__(self,plan,n256=False,cap=200):
        self.p=plan.p;self.original=plan.b7;self.splits=plan.b7.splits
        root=Path(__file__).resolve().parent
        body=mask_stage(plan.b7.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('__launch_bounds__(256,2)',f'__maxnreg__({cap})')
        point='if(threadIdx.x<128){\n  setmaxnreg_dec<32>();\n  if(threadIdx.x==0){'
        assert body.count(point)==1
        body=body.replace(point,'if(threadIdx.x>=128){\n  if(threadIdx.x==128){')
        point='}else{setmaxnreg_inc<224>();compute_source(p);}'
        assert body.count(point)==1;body=body.replace(point,'}else{compute_source(p);}')
        if n256:
            body=body.replace('TMN_DEVI void compute_source', (root/'mma256.cuh').read_text()+'\nTMN_DEVI void compute_source')
            point='float dw0[64]={},dw1[64]={};';assert body.count(point)==1
            body=body.replace(point,'float dw[128]={};')
            body=body.replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
            point='mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);'
            assert body.count(point)==1
            body=body.replace(point,'mma256<0,1>(dw,smem_desc(smem_u32(sm+DERIV+k*32),16,1024,1),smem_desc(smem_u32(xn+k*2048),8192,1024,1),it>0||k>0);')
            point='mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
            assert body.count(point)==1;body=body.replace(point,'')
            point=' for(int j=0;j<64;++j){';assert body.count(point)==1
            body=body.replace(point,' static_for<128>([&](auto jj){constexpr int j=decltype(jj)::value;')
            point='  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];\n }'
            assert body.count(point)==1;body=body.replace(point,'  p.part[ix]=dw[j];\n });')
        body=body.replace('mw_d256_b7_tma','mw_d256_capped_loader_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_capped_loader_source').kernel('mw_d256_capped_loader_source');self.smem=115072
        self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((32*self.splits,1,1),(160,1,1),[self.original.params],self.smem)
