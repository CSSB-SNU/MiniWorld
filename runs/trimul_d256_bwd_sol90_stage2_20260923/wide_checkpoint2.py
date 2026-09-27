"""Wide candidate with forward product saves and exact FP32 dWproj splits."""
import ctypes,os
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_combined import WideTraining
from wide_saved_products import SavedWideProductsOutput
from wide_split_dwp import SplitProjectionWeight


class Training(WideTraining):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.product_output=SavedWideProductsOutput(self.f,self.p,products=(self.b1.proj,self.b1.gate))
        self.artifacts.append(self.product_output.cubin)
        self.split_dwp=SplitProjectionWeight(self.p,self.b1.exact_dwp) if self.p.D==512 and self.p.n==768 else None
        self.ln=None
        threads=int(os.environ.get('CHECKPOINT_LN_THREADS','0'))
        if threads:
            from wide_bounded_ln import BoundedLN
            self.ln=BoundedLN(self.p,self.b1,threads)
            self.artifacts.append(self.ln.cubin)

    def forward(self,saved=True):
        if not saved:return super().forward(saved=False)
        self.f.mask.copy_(self.mask)
        self.p.mask.copy_(self.mask.reshape_as(self.p.mask))
        self.f.output=self.product_output
        return self.f()

    def backward(self):
        p=self.p;b=self.b1
        if p.D==512:
            b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem)
            torch.mm(p.tensors[6],b.wp.t(),out=b.proj)
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0)
        torch.mm(p.tensors[7],b.wp,out=p.tensors[9])
        if self.split_dwp is not None:self.split_dwp()
        elif b.exact_dwp is not None:b.exact_dwp()
        else:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
        torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg)
        if self.ln is not None:self.ln()
        else:
            L=T._launch_module();drv=b.ln.unit.drv;args=L._Packed([b.lp])
            drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(b.ln.handle)),b.grid,1,1,b.threads,1,1,b.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
        self.contract();self.b7.source_only();self.dx()
        return p.outputs
