"""Diagnostic FP32 accumulator witness for the bitwise full-K dX kernel."""
import torch
from d256_full_width_prefix_dx import FullWidthPrefixDX
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class Fp32DxWitness(FullWidthPrefixDX):
    def __init__(self,plan):
        super().__init__(plan,4,64,0,'256B')
        p=plan.p;body=self.source_text
        body=body.replace('CUtensorMap input,weight,output;','CUtensorMap input,weight,output;float* witness;')
        marker='  *reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n  size_t ix=size_t(blockIdx.x*128+WG*64+r)*256+c;p.witness[ix]=acc[j];p.witness[ix+1]=acc[j+1];')
        body=body.replace('mw_d256_full_width_prefix_dx','mw_d256_fp32_dx_witness')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=4','-DDX_FULL_K=64','-DDX_FULL_DEPTH=0']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_fp32_dx_witness').kernel('mw_d256_fp32_dx_witness');self.k.set_max_dynamic_smem(self.smem)
        self.witness=torch.empty((p.M,256),device=p.x.device,dtype=torch.float32)
        self.params=T._launch_module().Struct([*self.params.fields,self.witness])
