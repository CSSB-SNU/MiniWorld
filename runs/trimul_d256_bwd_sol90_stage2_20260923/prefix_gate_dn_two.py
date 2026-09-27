"""Gate epilogue and dNorm consume the same shared DP tile."""
from pathlib import Path
from prefix_gate_epi import PrefixGateEpi
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class PrefixGateDnTwo:
    def __init__(self,plan):
        p=plan.p;d=p.D;h=2*d;root=Path(__file__).resolve().parent
        prior=plan.prefix_gate
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}']
        if d!=512:flags.append('-DTMN_SIGMOID_TANH=1')
        body=(root/'prefix_gate_dn_two.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_prefix_gate_dn_two').kernel('mw_prefix_gate_dn_two');self.smem=40960+64*d*2+128
        self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module();tm=lambda t,rows,kr=64:launch.tensor_map(t,[64,kr],dims=[h,rows],strides_bytes=[h*2],swizzle='128B',l2='128B')
        fields=prior.params.fields.copy();fields[-2:-2]=[tm(plan.leaves[6] if d==256 else plan.b1.wp,d,32),tm(p.tensors[9],p.M)]
        self.params=launch.Struct(fields);self.grid=prior.grid
    def __call__(self):self.k.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
