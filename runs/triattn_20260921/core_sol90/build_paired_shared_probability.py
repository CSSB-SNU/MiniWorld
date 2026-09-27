from pathlib import Path
import hashlib,os,sys
from paired_shared_probability import transform
HERE=Path(__file__).resolve().parent
parent='m128r4coop3p2ip4v2q2';variant='coop4pairedp6qrv4'
two_cta='--two-cta' in sys.argv
if two_cta:variant='coop2pairedp6qr2ctav4'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
s=transform((HERE/(parent+'.cu')).read_text()).replace(parent,variant)
if two_cta:
 def replace(old,new,count=1):
  global s
  assert s.count(old)==count,(old,s.count(old),count)
  s=s.replace(old,new)
 replace('Rows=4,Stages=3','Rows=2,Stages=2')
 replace('Consumers=512,Threads=512','Consumers=256,Threads=256')
 replace('__launch_bounds__(Threads,1)','__launch_bounds__(Threads,2)')
 replace('sizeof(Shared)<=232448','sizeof(Shared)<=116224')
 replace('bias[4][64*N]','bias[2][64*N]');replace('bf[4]','bf[2]');replace('be[4]','be[2]')
 replace('for(int st=0;st<4;++st)','for(int st=0;st<2;++st)')
 replace('int st=seq%4;','int st=seq%2;')
 replace('if(seq>=4){s.be[st].wait(((seq/4)-1)&1);','if(seq>=2){s.be[st].wait(((seq/2)-1)&1);')
 replace('s.bf[seq%4].wait((seq/4)&1);','s.bf[seq%2].wait((seq/2)&1);')
 replace('s.bias[seq%4]','s.bias[seq%2]');replace('s.be[seq%4].arrive();','s.be[seq%2].arrive();',2)
 replace('s.be[(seq+1)%4].arrive();','s.be[(seq+1)%2].arrive();')
 replace('for(int seq=0;seq<4;++seq)load_bias(seq);','for(int seq=0;seq<2;++seq)load_bias(seq);')
 replace('seq+4<48)load_bias(seq+4);','seq+2<48)load_bias(seq+2);')
 replace('load_kv(0);load_kv(1);load_kv(2);','load_kv(0);load_kv(1);')
 replace('seq/4+3<12)load_kv(seq/4+3);','seq/4+2<12)load_kv(seq/4+2);')
 replace('2*pair+4<48){load_bias(2*pair+4);load_bias(2*pair+5);}',
         '2*pair+2<48){load_bias(2*pair+2);load_bias(2*pair+3);}')
 marker=' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);'
 replace(marker,''' int resident=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(resident>=2,"two resident CTAs required; got ",resident);
'''+marker)
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
