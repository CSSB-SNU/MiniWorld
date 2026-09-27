from pathlib import Path
import os,runpy,shutil
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=HERE/'build_stsm';BUILD.mkdir(exist_ok=True)
runpy.run_path(str(HERE/'stsm_variant.py'))
prepare=(HERE/'prepare.cu').read_text().replace('triattn_projected_prepare','triattn_projected_stsm_prepare').replace('projected_prepare_cuda','projected_stsm_prepare_cuda')
(HERE/'prepare_stsm.cu').write_text(prepare)
binding=(HERE/'binding.cpp').read_text().replace('projected_prepare_cuda','projected_stsm_prepare_cuda').replace('projected_qkv_cuda','projected_stsm_qkv_cuda')
(HERE/'binding_stsm.cpp').write_text(binding)
os.environ['MAX_JOBS']='2'
m=load(name='triattn_projected_stsm',sources=[str(HERE/n) for n in ('binding_stsm.cpp','prepare_stsm.cu','projection_stsm.cu')],
       extra_include_paths=[str(ROOT/'oc/opt_core/kernels/triattn_surround_tma'),str(ROOT.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
       extra_cflags=['-O3','-std=c++17'],
       extra_cuda_cflags=['-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a',
                         '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','--ptxas-options=-v'],
       build_directory=str(BUILD),verbose=True)
shutil.copy2(m.__file__,HERE/'triattn_projected_stsm.so')
print('BUILT',m.__file__,flush=True)
