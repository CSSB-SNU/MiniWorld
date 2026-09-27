from pathlib import Path
import os,runpy,shutil
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=HERE/'build';BUILD.mkdir(exist_ok=True)
runpy.run_path(str(HERE/'make_prepare.py'))
os.environ['MAX_JOBS']='2'
m=load(name='triattn_projected_proof',sources=[str(HERE/n) for n in ('binding.cpp','prepare.cu','projection.cu')],
       extra_include_paths=[str(ROOT/'oc/opt_core/kernels/triattn_surround_tma'),str(ROOT.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
       extra_cflags=['-O3','-std=c++17'],
       extra_cuda_cflags=['-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a',
                         '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','--ptxas-options=-v'],
       build_directory=str(BUILD),verbose=True)
shutil.copy2(m.__file__,HERE/'triattn_projected_proof.so')
print('BUILT',m.__file__,flush=True)
