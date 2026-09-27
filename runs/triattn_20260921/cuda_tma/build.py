from pathlib import Path
import os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
CUTLASS=HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include'
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
def build():
 (HERE/'build').mkdir(exist_ok=True)
 return load(name='triattn_surround_tma',sources=[str(HERE/'bindings.cpp'),str(HERE/'prologue.cu'),str(HERE/'epilogue.cu'),str(HERE/'prologue128.cu'),str(HERE/'prologue256.cu'),str(HERE/'epilogue128.cu'),str(HERE/'epilogue256.cu')],
  extra_include_paths=[str(CUTLASS)],extra_cflags=['-O3','-std=c++17'],
  extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v','--ftz=true'],
  extra_ldflags=['-lcuda'],build_directory=str(HERE/'build'),verbose=True)
if __name__=='__main__':build()
