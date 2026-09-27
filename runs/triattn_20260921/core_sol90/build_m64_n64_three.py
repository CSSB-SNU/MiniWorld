"""M64/N64, R2, 3 scores/2 P, independent K/V full barriers.

Distinct from the previous M128/N64 two-score/three-P design: one query
half per consumer, full Q in8 registers and a whole-body bias prefetch.
Producer32 + two consumers224 require an initial160-register CTA pool.
"""
from pathlib import Path
import hashlib,os,sys
HERE=Path(__file__).resolve().parent
variant='m64n64r2s3p2qrf40'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
s=(HERE/'m64r2s2p2qrf40.cu').read_text().replace('m64r2s2p2qrf40',variant)
s=s.replace('M=64,N=32,LN=128','M=64,N=64,LN=128')
s=s.replace('__launch_bounds__(Threads,2)','__launch_bounds__(Threads,1)')
s=s.replace('104>0','224>0').replace('warpgroup_reg_alloc<104>','warpgroup_reg_alloc<224>')
s=s.replace('qr,full[Stages][Rows],bf[2]','qr,full[Stages][Rows],vfull[Stages][Rows],bf[2]')
s=s.replace('s.full[st][r].init(2);','s.full[st][r].init(1);s.vfull[st][r].init(1);')
a=s.index('  if(tid==Consumers+64) {');b=s.index('  if(tid==Consumers+32) {',a)
s=s[:a]+s[a:b].replace('s.full[st][r]','s.vfull[st][r]')+s[b:]
s=s.replace('Score score;Score next_score;Output acc[1];','Score scores[3];auto& score=scores[0];auto& next_score=scores[1];Output acc[1];')
# Preserve the installed first32-key seed rather than seeding over all64.
s=s.replace('for(int col=1;col<N/4;col++)m=fmaxf(m,sr(row,col));','for(int col=1;col<(Safe?N/4:8);col++)m=fmaxf(m,sr(row,col));')
# Every PV first use acquires V independently from QK's K-ready wait.
old='''  constexpr int hh=0;
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(Ratio))%Stages][wg]),SV{});'''
assert s.count(old)==2
s=s.replace(old,'''  constexpr int hh=0;
  if(seq%Ratio==0){s.vfull[(seq/Ratio)%Stages][wg].wait(((seq/Ratio)/Stages)&1);asm volatile("":::"memory");}
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(Ratio))%Stages][wg]),SV{});''')
# All waits must keep the scalar/shared operations below retirement.
s=s.replace('warpgroup_wait<0>();warpgroup_fence_operand(score);','warpgroup_wait<0>();asm volatile("":::"memory");warpgroup_fence_operand(scores[2]);warpgroup_fence_operand(score);')
a=s.index(' }else{\n  clear(pp);clear(pp1);',s.index(' auto drain='))
b=s.index('\n auto id=pv.get_slice(t)',a)
body=''' }else{
  clear(pp);clear(pp1);
  init_score(scores[0],0);issue_qk(scores[0],0,_0{});
  init_score(scores[1],1);
  drain();
  auto step=[&](auto seqc) __attribute__((always_inline)) {
   constexpr int seq=decltype(seqc)::value;
   auto& sc=scores[seq%3];
   auto& ns=scores[(seq+1)%3];
   auto& old=scores[(seq+2)%3];
   auto& prob=(seq%2==0)?pp1:pp;
   if constexpr(seq>0){warpgroup_wait<1>();asm volatile("":::"memory");}
   warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   release_k(seq);
   if constexpr(seq>=3)release(seq-3);
   if constexpr(seq>0){pack(old,prob);warpgroup_fence_operand(prob);}
   if constexpr(seq+1<768/N)issue_qk(ns,seq+1,_0{});
   exponentiate(sc,_0{},cute::bool_constant<seq==0>{});
   if constexpr(seq==0){
    // The real zero-P dummy PV makes the next wait1 retire QK1. P1 will
    // not be overwritten until wait1 at step2 has retired this read.
    issue_pv(0,_0{},pp1);
   }else{
    if constexpr(seq+1<768/N)issue_pv_prefenced(seq-1,_0{},prob);
    else issue_pv(seq-1,_0{},prob);
   }
   if constexpr(seq+2<768/N)init_score(old,seq+2);
  };
STEPS
  drain();
  release(768/N-3);release(768/N-2);
  pack(scores[(768/N-1)%3],pp1);
  issue_pv(768/N-1,_0{},pp1);
  drain();release(768/N-1);
 }
'''
body=body.replace('STEPS','\n'.join('  step(Int<%d>{});'%i for i in range(12)))
s=s[:a]+body+s[b:]
s=s.replace('TORCH_CHECK(resident>=2,"two-CTA occupancy required, got ",resident," smem=",sizeof(Shared));','TORCH_CHECK(resident==1,"one-CTA occupancy required, got ",resident," smem=",sizeof(Shared));')
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/'m64r2s2p2qrf40.cpp').read_text().replace('m64r2s2p2qrf40',variant))
if '--generate-only' in sys.argv:print('GENERATED',variant);sys.exit(0)
from torch.utils.cpp_extension import load
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],
 extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],
 extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','--keep','--keep-dir='+str(directory)],
 extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
