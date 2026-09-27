"""Channel-major saved preactivations, coalesced packed global GP."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SavedGpLinear:
    def __init__(self,original,pre,mask,grid_factor=8):
        self.original=original;self.p=p=original.p
        body=Path(__file__).with_suffix('.cu').read_text()
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_saved_gp_linear').kernel('mw_wide_saved_gp_linear')
        self.params=T._launch_module().Struct([pre,p.dl,p.dr,mask,*p.gp,p.M])
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*grid_factor
    def derivatives(self):self.k.launch((self.grid,1,1),(256,1,1),[self.params],0)
    def __call__(self):self.derivatives();self.original.matmul()
