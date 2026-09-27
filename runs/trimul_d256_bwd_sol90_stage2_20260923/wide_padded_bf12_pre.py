"""Store each 32 BF16 values as a contiguous 48-byte lossless block."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PaddedBF12Pre:
    def __init__(self,pre):
        self.pre=pre
        assert pre.numel()%32==0
        self.storage=torch.empty((pre.numel()//32,48),device=pre.device,dtype=torch.uint8)
        self.escapes=torch.zeros((),device=pre.device,dtype=torch.int64)
        body=Path(__file__).with_name('pack_pre_bf12.cu').read_text()
        body=body.replace('uint16_t* mant;uint8_t* exps;','uint8_t* packed;')
        body=body.replace('  p.mant[i]=','  size_t base=size_t(i/16)*48;int local=i%16;\n  *reinterpret_cast<uint16_t*>(p.packed+base+local*2)=')
        body=body.replace('p.exps[i]=','p.packed[base+32+local]=')
        body=body.replace('mw_pack_pre_bf12','mw_pack_padded_bf12')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_pack_padded_bf12').kernel('mw_pack_padded_bf12')
        self.params=T._launch_module().Struct([pre,self.storage,self.escapes,pre.numel()//2])
    def __call__(self):
        self.escapes.zero_();self.k.launch((1056,1,1),(256,1,1),[self.params],0)
