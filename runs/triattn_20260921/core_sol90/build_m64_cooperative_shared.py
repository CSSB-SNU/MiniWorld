from pathlib import Path
import hashlib,os,sys
from m64_cooperative_shared import transform
HERE=Path(__file__).resolve().parent
parent='m128r4coop3p2ip4v2q2';variant='m64coop2s4p3qr'
deep_kv='--deep-kv' in sys.argv
if deep_kv:variant+='kv4b2'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
s=transform((HERE/(parent+'.cu')).read_text()).replace(parent,variant)
if deep_kv:
 def replace(old,new,count=1):
  global s
  assert s.count(old)==count,(old,s.count(old),count)
  s=s.replace(old,new)
 replace('Rows=2,Stages=3','Rows=2,Stages=4')
 replace('bias[4][64*N]','bias[2][64*N]');replace('bf[4]','bf[2]');replace('be[4]','be[2]')
 replace('for(int st=0;st<4;++st)','for(int st=0;st<2;++st)')
 replace('int st=seq%4;','int st=seq%2;')
 replace('if(seq>=4){s.be[st].wait(((seq/4)-1)&1);','if(seq>=2){s.be[st].wait(((seq/2)-1)&1);')
 replace('s.bf[seq%4].wait((seq/4)&1);','s.bf[seq%2].wait((seq/2)&1);')
 replace('s.bias[seq%4]','s.bias[seq%2]');replace('s.be[seq%4].arrive();','s.be[seq%2].arrive();')
 replace('for(int seq=0;seq<4;++seq)load_bias(seq);','for(int seq=0;seq<2;++seq)load_bias(seq);')
 replace('load_kv(0);load_kv(1);load_kv(2);','load_kv(0);load_kv(1);load_kv(2);load_kv(3);')
 replace('seq/2+3<12)load_kv(seq/2+3);','seq/2+4<12)load_kv(seq/2+4);')
 replace('if(tid==0 && seq+4<24)load_bias(seq+4);','if(tid==0 && seq+2<24)load_bias(seq+2);',2)
 # SAFE replenishes its just-consumed slot; hot already issued QK(k+1)
 # in the preceding body, so it may refill that slot with bias(k+3).
 a=s.index('  }else{\n   clear(sc[3]);')
 hot=s[a:]
 assert hot.count('if(tid==0 && seq+2<24)load_bias(seq+2);')==1
 hot=hot.replace('if(tid==0 && seq+2<24)load_bias(seq+2);','if(tid==0 && seq+3<24)load_bias(seq+3);')
 s=s[:a]+hot
 replace('init(sc[1],1);issue_qk(sc[1],1,_0{});init(sc[2],2);drain();',
         'init(sc[1],1);issue_qk(sc[1],1,_0{});if(tid==0)load_bias(2);init(sc[2],2);drain();')
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
