from pathlib import Path
import os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
CUTLASS=HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include'
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
def build():
 (HERE/'build_epi').mkdir(exist_ok=True)
 return load(name='triattn_cta_epilogue',sources=[str(HERE/'bindings_epi.cpp'),str(HERE/'epilogue_pipeline.cu'),str(HERE/'epilogue_pipeline128.cu'),str(HERE/'epilogue_split.cu')],
  extra_include_paths=[str(CUTLASS)],extra_cflags=['-O3','-std=c++17'],
  extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v','--ftz=true'],
  extra_ldflags=['-lcuda'],build_directory=str(HERE/'build_epi'),verbose=True)
if __name__=='__main__':build()
