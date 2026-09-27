"""M64/N64 R3, two scores/two P, full Q, independent K/V readiness.

The R2 three-score profile underutilizes execution units. Keeping current
E-to-PV in the same body saves32 score registers for a third consumer WG.
"""
from pathlib import Path
import hashlib,os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
variant='m64n64r3s2p2qrf40'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
s=(HERE/'m64r2s2p2qrf40.cu').read_text().replace('m64r2s2p2qrf40',variant)
s=s.replace('M=64,N=32,LN=128','M=64,N=64,LN=128').replace('Rows=2,Stages=2','Rows=3,Stages=2')
s=s.replace('__launch_bounds__(Threads,2)','__launch_bounds__(Threads,1)')
s=s.replace('104>0','160>0').replace('warpgroup_reg_alloc<104>','warpgroup_reg_alloc<160>')
s=s.replace('qr,full[Stages][Rows],bf[2]','qr,full[Stages][Rows],vfull[Stages][Rows],bf[2]')
s=s.replace('s.full[st][r].init(2);','s.full[st][r].init(1);s.vfull[st][r].init(1);')
a=s.index('  if(tid==Consumers+64) {');b=s.index('  if(tid==Consumers+32) {',a)
s=s[:a]+s[a:b].replace('s.full[st][r]','s.vfull[st][r]')+s[b:]
s=s.replace('for(int col=1;col<N/4;col++)m=fmaxf(m,sr(row,col));','for(int col=1;col<(Safe?N/4:8);col++)m=fmaxf(m,sr(row,col));')
old='''  constexpr int hh=0;
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(Ratio))%Stages][wg]),SV{});'''
assert s.count(old)==2
s=s.replace(old,'''  constexpr int hh=0;
  if(seq%Ratio==0){s.vfull[(seq/Ratio)%Stages][wg].wait(((seq/Ratio)/Stages)&1);asm volatile("":::"memory");}
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(Ratio))%Stages][wg]),SV{});''')
s=s.replace('warpgroup_wait<0>();warpgroup_fence_operand(score);','warpgroup_wait<0>();asm volatile("":::"memory");warpgroup_fence_operand(score);')
s=s.replace('warpgroup_wait<1>();warpgroup_fence_operand(sc);','warpgroup_wait<1>();asm volatile("":::"memory");warpgroup_fence_operand(sc);')
# The static per-step index makes real tail elimination compile-time.
s=s.replace('auto step=[&](int seq,auto which,auto seed_possible) __attribute__((always_inline)) {','auto step=[&](auto seqc,auto which,auto seed_possible) __attribute__((always_inline)) {\n   constexpr int seq=decltype(seqc)::value;')
s=s.replace('   issue_qk(ns,seq+1,_0{});','   if constexpr(seq+1<768/N)issue_qk(ns,seq+1,_0{});')
s=s.replace('   init_score(sc,seq+2);','   if constexpr(seq+2<768/N)init_score(sc,seq+2);')
a=s.index('  step(0,_0{},cute::true_type{});');b=s.index('  drain();release(nk-1);',a)
s=s[:a]+'\n'.join('  step(Int<%d>{},_%d{},cute::%s_type{});'%(i,i%2,'true' if i==0 else 'false') for i in range(12))+'\n'+s[b:]
# Match the installed hot validation's finite-numerator requirement.
s=s.replace(' bool bad=missing_seed;',''' bool bad=missing_seed;
 if constexpr(!Safe){
  #pragma unroll
  for(int j=0;j<16;++j)bad|=(__float_as_uint(acc[0](j))&0x7f800000u)==0x7f800000u;
 }''')
s=s.replace('TORCH_CHECK(resident>=2,"two-CTA occupancy required, got ",resident," smem=",sizeof(Shared));','TORCH_CHECK(resident==1,"one-CTA occupancy required, got ",resident," smem=",sizeof(Shared));')
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/'m64r2s2p2qrf40.cpp').read_text().replace('m64r2s2p2qrf40',variant))
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','--keep','--keep-dir='+str(directory)],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
