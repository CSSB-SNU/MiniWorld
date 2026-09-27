from pathlib import Path
import os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
def build():
 (HERE/'build_pipe2').mkdir(exist_ok=True)
 return load(name='triattn_core_pipe2',sources=[str(HERE/'bindings_pipe2.cpp'),str(HERE/'core_pipe2.cu')],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v'],extra_ldflags=['-lcuda'],build_directory=str(HERE/'build_pipe2'),verbose=True)
if __name__=='__main__':build()
