from pathlib import Path
import os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
def build():
 (HERE/'build_mask').mkdir(exist_ok=True)
 return load(name='triattn_mask_stage',sources=[str(HERE/'mask_stage.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(HERE/'build_mask'),verbose=True)
if __name__=='__main__':build()
