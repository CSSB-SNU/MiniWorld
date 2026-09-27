"""Emit the output-gate gradient directly into the ordered dX prefix."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class PrefixGateEpi:
    def __init__(self,plan,grid_factor=4):
        p=plan.p;d=p.D;self.p=p
        b=plan.b1 if d!=256 else plan.b1.prepare.original
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}']
        if d!=512:flags.append('-DTMN_SIGMOID_TANH=1')
        self.cubin=T.compile(Path(__file__).with_name('prefix_gate_epi.cu'),flags)
        self.k=T.load_unit(str(self.cubin),'mw_prefix_gate_epi').kernel('mw_prefix_gate_epi')
        self.smem=49152+128;self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module()
        def tm(x,rows):return launch.tensor_map(x,[64,64],dims=[d,rows],strides_bytes=[d*2],swizzle='128B',l2='128B')
        self.prefix=plan.dx.input[:d]
        dg=launch.tensor_map(self.prefix,[64,64],dims=[p.M,d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        self.params=launch.Struct([tm(b.proj,p.M),tm(b.gate,p.M),tm(p.dy,p.M),tm(p.ds,p.n),tm(p.tensors[7],p.M),dg,p.M,p.n])
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*grid_factor
    def __call__(self):self.k.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
