from pathlib import Path
import os
from torch.utils.cpp_extension import load
from cooperative_four_score import transform
HERE=Path(__file__).resolve().parent
qr=os.environ.get('COOP_QREGS')=='1'
variant='coop3s4qr' if qr else 'coop3s4ss'
opaque=os.environ.get('COOP_OPAQUE')=='1'
if opaque: variant+='z'
full=os.environ.get('COOP_UNROLL')=='1'
if full: variant+='full'
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
s=transform(qr)
if full:
 a=s.index('   #pragma unroll 1\n   for(int period=0;period<3;++period){')
 b=s.index('   pack(sc[3],pp[1]);',a)
 body=[]
 for period in range(3):
  for d in range(16 if period<2 else 14):
   body.append('   step(%d,Int<%d>{});'%(period,d))
  body.append('   drain();rescale(sc[%d]);'%(1 if period<2 else 3))
 s=s[:a]+'\n'.join(body)+'\n'+s[b:]
if opaque:
 s=s.replace('struct Shared {','struct Shared {\n int zero;')
 s=s.replace(' if(tid==0){\n  s.qr.init(1);',' if(tid==0){\n  s.zero=0;\n  s.qr.init(1);')
 s=s.replace('auto kb=tq.partition_fragment_B(ks);','auto kr=tq.partition_fragment_B(ks);auto kd=kr.data().desc_;kd.reg32_[0]+=uint32_t(*reinterpret_cast<int volatile*>(&s.zero));\n   auto kb=make_tensor(GMMA::DescriptorIterator{kd},kr.layout());')
 s=s.replace('auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());','desc.reg32_[0]+=uint32_t(*reinterpret_cast<int volatile*>(&s.zero));\n   auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());')
s=s.replace('SOL_NAMESPACE','ta_sol_'+variant).replace('sol_forward','sol_forward_'+variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp')
cpp.write_text('#include <torch/extension.h>\nat::Tensor sol_forward_'+variant+'(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_'+variant+');}\n')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
