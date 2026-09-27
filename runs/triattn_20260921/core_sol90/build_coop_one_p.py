from pathlib import Path
import os
from torch.utils.cpp_extension import load
from cooperative_one_p import transform
HERE=Path(__file__).resolve().parent
rows=int(os.environ.get('COOP_ROWS','5'))
variant='coop%ds2p1'%rows
full=os.environ.get('COOP_UNROLL')=='1'
ip=os.environ.get('COOP_IP4')=='1'
if full: variant+='full'
if ip: variant+='ip4'
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
s=transform(rows)
if full:
 a=s.index('   #pragma unroll 1\n   for(int period=0;period<3;++period){')
 b=s.index('   release(47);',a)
 body=[]
 for period in range(3):
  for d in range(16):
   seq=period*16+d
   body.append('   step(%d,Int<%d>{},cute::%s_type{});'%(seq,d&1,'true' if seq<2 else 'false'))
  body.append('   drain();rescale(sc[1]);')
 s=s[:a]+'\n'.join(body)+'\n'+s[b:]
if ip:
 old='''    #pragma unroll
    for(int col=0;col<8;++col)sr(row,col)=ex2(fmaf(sr(row,col),c,nm[hh][row]));'''
 assert old in s
 s=s.replace(old,'''    #pragma unroll
    for(int col=0;col<8;col+=4){
     asm volatile("fma.rn.ftz.f32 %0,%0,%4,%5;\\n"
                  "fma.rn.ftz.f32 %1,%1,%4,%5;\\n"
                  "fma.rn.ftz.f32 %2,%2,%4,%5;\\n"
                  "fma.rn.ftz.f32 %3,%3,%4,%5;\\n"
                  "ex2.approx.ftz.f32 %0,%0;\\n"
                  "ex2.approx.ftz.f32 %1,%1;\\n"
                  "ex2.approx.ftz.f32 %2,%2;\\n"
                  "ex2.approx.ftz.f32 %3,%3;\\n"
                  :"+f"(sr(row,col)),"+f"(sr(row,col+1)),"+f"(sr(row,col+2)),"+f"(sr(row,col+3))
                  :"f"(c),"f"(nm[hh][row]));
    }''')
s=s.replace('SOL_NAMESPACE','ta_sol_'+variant).replace('sol_forward','sol_forward_'+variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp')
cpp.write_text('#include <torch/extension.h>\nat::Tensor sol_forward_'+variant+'(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_'+variant+');}\n')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
