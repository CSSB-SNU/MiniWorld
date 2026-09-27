"""Pilot FP32 batched output-weight GEMM retaining all 32 split intervals."""
import torch
from lt_contract import LtBmm


class SplitProjectionWeight:
    def __init__(self, p, native):
        self.p = p
        self.native = native
        d, h, step = p.D, 2*p.D, p.M//32
        self.partial = p.floats[7].as_strided((32,d,h),(11*d*d,h,1))
        a = p.tensors[7].as_strided((32,d,step),(step*d,1,d))
        b = p.tensors[6].as_strided((32,step,h),(step*h,h,1))
        self.workspace = torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.matmul = LtBmm(a,b,self.partial,self.workspace)

    def __call__(self):
        self.matmul()
        self.native.reduce.launch((264,1,1),(256,1,1),[self.p.params],0)
