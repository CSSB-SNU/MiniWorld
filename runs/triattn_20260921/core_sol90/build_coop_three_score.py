"""Four cooperative WGs with 3 scores / 2 P, QK one chunk ahead.

Unlike the old two-score design, bias LDS is prefetched a full body before
QK and there is just one wait1 per body. P(k-1) is packed after PV(k-3)
retires; E(k) stays in its score buffer until the following step.
"""
from pathlib import Path
import hashlib,os,sys
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
mode=sys.argv[1] if len(sys.argv)>1 else 'normal'
assert mode in ('normal','q2','ru0','epack','q2epack')
parent='m128r4coop2f2';variant='m128r4coop3p2ip4v2'+('' if mode=='normal' else mode)
s=(HERE/(parent+'.cu')).read_text().replace(parent,variant)
s=s.replace('Score sc[2];','Score sc[3];')
s=s.replace('for(int n=0;n<2;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<3;++n)warpgroup_fence_operand(sc[n]);')
old='''    #pragma unroll
    for(int col=0;col<8;++col)sr(row,col)=ex2(fmaf(sr(row,col),c,nm[hh][row]));'''
assert s.count(old)==1
s=s.replace(old,'''    if constexpr(!Safe){
     #pragma unroll
     for(int col=0;col<8;col+=4){
      asm volatile(
       "fma.rn.ftz.f32 %0,%0,%4,%5;\\n"
       "fma.rn.ftz.f32 %1,%1,%4,%5;\\n"
       "fma.rn.ftz.f32 %2,%2,%4,%5;\\n"
       "fma.rn.ftz.f32 %3,%3,%4,%5;\\n"
       "ex2.approx.ftz.f32 %0,%0;\\n"
       "ex2.approx.ftz.f32 %1,%1;\\n"
       "ex2.approx.ftz.f32 %2,%2;\\n"
       "ex2.approx.ftz.f32 %3,%3;\\n"
       : "+f"(sr(row,col)),"+f"(sr(row,col+1)),"+f"(sr(row,col+2)),"+f"(sr(row,col+3))
       : "f"(c),"f"(nm[hh][row]));
     }
    }else{
'''+old+'''
    }''')
a=s.index('  }else{\n   init(sc[0],0);')
b=s.index('\n  auto id=',a)
calls='\n'.join('    step(period*24+%d,Int<%d>{},Int<%d>{},cute::%s_type{});'%(i,i&1,i%3,'true' if i<2 else 'false') for i in range(24))
s=s[:a]+'''  }else{
   clear(sc[2]);
   init(sc[0],0);issue_qk(sc[0],0,_0{});init(sc[1],1);drain();
   auto step=[&](int seq,auto hc,auto sic,auto seed_possible) __attribute__((always_inline)){
    constexpr int hh=decltype(hc)::value,si=decltype(sic)::value;
    auto& cur=sc[si];auto& future=sc[(si+1)%3];auto& old=sc[(si+2)%3];
    auto& previous=pp[1-hh];
    // Retire QK(k) and PV(k-3); PV(k-2) may remain. P(k-1) recycles
    // the slot formerly occupied by P(k-3), so no second wait is needed.
    warpgroup_wait<1>();warpgroup_fence_operand(cur);warpgroup_fence_operand(previous);
    if(seq>=3)release(seq-3);
    if(tid==0 && seq+8<48)load_bias(seq+8);
    pack(old,previous);warpgroup_fence_operand(previous);
    issue_qk(future,seq+1,Int<1-hh>{});
    if constexpr(decltype(seed_possible)::value){
     if(seq<2)exponentiate(cur,hc,cute::true_type{});else exponentiate(cur,hc,cute::false_type{});
    }else exponentiate(cur,hc,cute::false_type{});
    // The first zero-P PV is also a required second committed group:
    // next step's wait1 must retire QK(1), not leave it as the newest group.
    issue_pv(previous,max(seq-1,0),Int<1-hh>{},cute::true_type{});
    init(old,seq+2);
   };
   #pragma unroll 1
   for(int period=0;period<2;++period){
'''+calls+'''
    drain();rescale(sc[2]);
   }
   pack(sc[2],pp[1]);issue_pv(pp[1],47,_1{},cute::false_type{});drain();release(47);
  }
'''+s[b:]
if mode.startswith('q2'):
 old='init(sc[0],0);issue_qk(sc[0],0,_0{});init(sc[1],1);drain();'
 assert s.count(old)==1
 s=s.replace(old,'init(sc[0],0);issue_qk(sc[0],0,_0{});init(sc[1],1);issue_qk(sc[1],1,_1{});drain();')
 old='warpgroup_wait<1>();warpgroup_fence_operand(cur);warpgroup_fence_operand(previous);'
 assert s.count(old)==1
 s=s.replace(old,'warpgroup_wait<2>();warpgroup_fence_operand(cur);warpgroup_fence_operand(previous);')
 old='    issue_qk(future,seq+1,Int<1-hh>{});'
 assert s.count(old)==1
 s=s.replace(old,'    init(old,seq+2);issue_qk(old,seq+2,hc);')
 old='    init(old,seq+2);\n   };'
 assert s.count(old)==1
 s=s.replace(old,'   };')
 s=s.replace('// Retire QK(k) and PV(k-3); PV(k-2) may remain.',
             '// Retire QK(k) and PV(k-3); QK(k+1) and PV(k-2) may remain.')
if mode.endswith('epack'):
 old='   warpgroup_fence_operand(a);\n  };\n  auto pack='
 assert s.count(old)==1
 s=s.replace(old,'''   if constexpr(!Safe){
    // Keep completed E as eight BF16 pairs until next body's P reuse.
    // Low-to-high compaction never overwrites an unread source element.
    #pragma unroll
    for(int pair=0;pair<8;++pair){
     auto packed=__floats2bfloat162_rn(a(2*pair),a(2*pair+1));
     a(pair)=__uint_as_float(reinterpret_cast<uint32_t const&>(packed));
    }
    #pragma unroll
    for(int pair=8;pair<16;++pair)a(pair)=0.f;
   }
'''+old)
 old='for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}'
 assert s.count(old)==1
 s=s.replace(old,'''for(int n=0;n<8;++n){
    if constexpr(!Safe)dst(n)=__float_as_uint(a(n));
    else {auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}
   }''')
 old='''       #pragma unroll
       for(int col=0;col<8;++col)pr(row,col)*=f;'''
 assert s.count(old)==1
 s=s.replace(old,'''       // Isolated numeric experiment: rescale rounded pending P.
       // This is not a proof of equality near BF16 under/overflow.
       #pragma unroll
       for(int pair=0;pair<4;++pair){
        int index=2*pair+row;uint32_t bits=__float_as_uint(pend(index));
        auto packed=*reinterpret_cast<__nv_bfloat162 const*>(&bits);
        float2 v=__bfloat1622float2(packed);
        auto scaled=__floats2bfloat162_rn(v.x*f,v.y*f);
        pend(index)=__uint_as_float(reinterpret_cast<uint32_t const&>(scaled));
       }''')
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/(parent+'.cpp')).read_text().replace(parent,variant))
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED']+(['-Xptxas=--register-usage-level=0'] if mode=='ru0' else []),extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
