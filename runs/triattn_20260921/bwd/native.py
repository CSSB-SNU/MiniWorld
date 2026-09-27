from pathlib import Path
import os
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
os.environ['MAX_JOBS']='2'
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
EXT=None
def extension():
    global EXT
    if EXT is None:
        build=ROOT/'build';build.mkdir(exist_ok=True)
        EXT=load(name='triattn_projection_bwd',sources=[str(ROOT/'projection_dgrad.cu')],
            build_directory=str(build),
            extra_include_paths=[str(ROOT.parent/'oc/opt_core/kernels/triattn_core_broadcast/csrc'),str(ROOT.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v'],verbose=True)
    return EXT
if __name__=='__main__':extension()
