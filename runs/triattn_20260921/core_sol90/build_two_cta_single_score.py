from pathlib import Path
import os
from torch.utils.cpp_extension import load
from two_cta_single_score import transform
HERE=Path(__file__).resolve().parent
variant='m128r2s1p1twocta'
period_size=int(os.environ.get('TWOCTA_PERIOD','16'))
if period_size!=16:variant+='p'+str(period_size)
pv_fence=os.environ.get('TWOCTA_FENCE')=='1'
if pv_fence:variant+='f'
scalar=os.environ.get('TWOCTA_SCALAR')=='1'
if scalar:variant+='sc'
den_mma=os.environ.get('TWOCTA_DENMMA')=='1'
if den_mma:
 assert scalar and pv_fence
 variant+='mma'
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
s=transform(period_size,pv_fence)
if scalar:
 from two_cta_scalar_den import transform as scalar_den
 s=scalar_den(s,den_mma)
s=s.replace('SOL_NAMESPACE','ta_sol_'+variant).replace('sol_forward','sol_forward_'+variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp')
cpp.write_text('#include <torch/extension.h>\nat::Tensor sol_forward_'+variant+'(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_'+variant+');}\n')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
