"""Bounded 96-register producer proof, separate from the 64-register artifact."""
from pathlib import Path
import os,shutil
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BUILD=HERE/'build_stsm96';BUILD.mkdir(exist_ok=True)
for src,dst in [('projection_stsm.cu','projection_stsm96.cu'),('prepare_stsm.cu','prepare_stsm96.cu'),('binding_stsm.cpp','binding_stsm96.cpp')]:
 s=(HERE/src).read_text().replace('triattn_projected_stsm','triattn_projected_stsm96').replace('projected_stsm_','projected_stsm96_')
 s=s.replace('__launch_bounds__(128,8)','__launch_bounds__(128,5)')
 (HERE/dst).write_text(s)
os.environ['MAX_JOBS']='2'
m=load(name='triattn_projected_stsm96',sources=[str(HERE/n) for n in ('binding_stsm96.cpp','prepare_stsm96.cu','projection_stsm96.cu')],
 extra_include_paths=[str(ROOT/'oc/opt_core/kernels/triattn_surround_tma'),str(ROOT.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
 extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','--ptxas-options=-v'],
 build_directory=str(BUILD),verbose=True)
shutil.copy2(m.__file__,HERE/'triattn_projected_stsm96.so')
print('BUILT',m.__file__,flush=True)
