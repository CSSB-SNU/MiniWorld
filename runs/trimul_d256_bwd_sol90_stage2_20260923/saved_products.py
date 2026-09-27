"""Save our K3 projection/gate outputs and skip their backward GEMMs."""
from saved_norm import SavedNormOutput
import os
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SavedProductsPrepare:
    def __init__(self, original, grid=528, threads=256):
        self.original=original
        self.grid,self.threads=grid,threads
        assert grid>0 and threads in (128,256)
        self.k=original.epi
        if threads==128:
            source=Path(__file__).with_name('norm.cu').read_text().split('struct Epi ')[1]
            body='#include "tmn_kernels.cuh"\nusing namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;\nstruct Epi '+source
            body=body.replace('mw_d256_gate_grad','mw_d256_gate_grad_flexible')
            body=body.replace('blockIdx.x*256','blockIdx.x*blockDim.x').replace('gridDim.x*256','gridDim.x*blockDim.x')
            flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DTMN_SIGMOID_TANH=1','-DMW_MINB=1','-DMW_K1_STREAM=0']
            out=T.compile_text(body,flags)
            self.k=T.load_unit(str(out),'mw_d256_gate_grad_flexible').kernel('mw_d256_gate_grad_flexible')

    def __call__(self):
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.original.ep],0)


def enable(plan):
    original=plan.b1.prepare
    assert hasattr(original,'proj') and hasattr(original,'gate')
    plan.f.output=SavedNormOutput(plan.f,plan.p,products=(original.proj,original.gate))
    plan.b1.prepare=SavedProductsPrepare(original,int(os.environ.get('SAVE_EPI_GRID','528')),int(os.environ.get('SAVE_EPI_THREADS','256')))
    plan.saved=(*plan.saved,original.proj,original.gate)
    return dict(name='saved_products',output_cubin=str(plan.f.output.cubin),
                saved_bytes=original.proj.numel()*original.proj.element_size()*2,
                epi_grid=plan.b1.prepare.grid,epi_threads=plan.b1.prepare.threads)
