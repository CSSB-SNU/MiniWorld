from pathlib import Path
import hashlib,os,sys
from m64_cooperative_register import transform
HERE=Path(__file__).resolve().parent
parent='m128r4coop3p2ip4v2q2';variant='m64coop2s4p2sskv4'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
s=transform((HERE/(parent+'.cu')).read_text()).replace(parent,variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/(parent+'.cpp')).read_text().replace(parent,variant))
if '--generate-only' in sys.argv:
 print('GENERATED',variant,len(s));sys.exit(0)
from torch.utils.cpp_extension import load
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','--keep-dir='+str(directory),'-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
