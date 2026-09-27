from pathlib import Path
import os
from torch.utils.cpp_extension import load
from r4_warp import transform
HERE=Path(__file__).resolve().parent
cooperative=os.environ.get('R4_COOPERATIVE')=='1'
variant='m128r4coop2' if cooperative else 'm128r4wp2'
fencefix=os.environ.get('R4_FENCEFIX','')
if fencefix: variant+='f'+fencefix
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
s=transform()
if cooperative:
 from r4_cooperative import transform as coop
 s=coop(s)
if fencefix:
 old='    pack(sc[1],pp[1]);\n   }'
 assert old in s
 s=s.replace(old,'    warpgroup_fence_operand(sc[1]);\n    pack(sc[1],pp[1]);\n    warpgroup_fence_operand(pp[1]);\n    warpgroup_arrive();\n   }')
 if fencefix=='2':
  old='if constexpr(!decltype(prefenced)::value){warpgroup_fence_operand(prob);warpgroup_arrive();}'
  assert old in s
  s=s.replace(old,'warpgroup_fence_operand(prob);warpgroup_arrive();')
s=s.replace('SOL_NAMESPACE','ta_sol_'+variant).replace('sol_forward','sol_forward_'+variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp')
cpp.write_text('#include <torch/extension.h>\nat::Tensor sol_forward_'+variant+'(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_'+variant+');}\n')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
