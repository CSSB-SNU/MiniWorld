"""Fuse contraction results directly into saved-preactivation GP, without dL/dR writes."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ContractGP:
    def __init__(self,plan,fused=True,transposed=False):
        p=plan.p;d=p.D;h=2*d;n=p.n;self.p=p
        root=Path(__file__).resolve().parent
        body=(root/'wide_contract_gp.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DFUSED_GP={int(fused)}',f'-DPRE_TRANSPOSED={int(transposed)}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_contract_gp').kernel('mw_wide_contract_gp')
        self.smem=32768+128;self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module();ab=p.front.ab
        def tm(x):return launch.tensor_map(x,[64,64],dims=[n,d*n],strides_bytes=[n*2],swizzle='128B',l2='128B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=launch.Struct([*[tm(x) for x in aa],*[tm(x) for x in bb],plan.pre,plan.f.mask,*p.gp,p.dl,p.dr,n])
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//64)**2,1,1),(128,1,1),[self.params],self.smem)
