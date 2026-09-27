"""M128/N16 R4: two live stream scores, full Q, PV then future QK.

Bootstrap retains four scores only to preserve first32-key seeds. Thereafter
wait1 leaves QK(k+1) outstanding while E(k) runs. Two P buffers are retained.
"""
from pathlib import Path
import os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
old='m128n16r4s4p2qrf40';name='m128n16r4s2p2qrf40'
s=(HERE/(old+'.cu')).read_text().replace(old,name)
# Dead bootstrap temporaries must not be kept live by final operand fences.
anchor='for(int j=0;j<4;++j)warpgroup_fence_operand(scores[j]);'
assert s.count(anchor)==1
s=s.replace(anchor,'for(int j=0;j<2;++j)warpgroup_fence_operand(scores[j]);')
a=s.index(' }else{\n  // Four scores, two P fragments;')
b=s.index('\n auto id=pv.get_slice(t)',a)
bootstrap=s[a:b]
seed=bootstrap[bootstrap.index('  auto seed_pair='):bootstrap.index('  exponentiate(scores[0]')]
body=''' }else{
  clear(pp);clear(pp1);
  init_score(scores[0],0);issue_qk(scores[0],0,_0{});
  init_score(scores[1],1);issue_qk(scores[1],1,_1{});
  init_score(scores[2],2);issue_qk(scores[2],2,_0{});
  init_score(scores[3],3);issue_qk(scores[3],3,_1{});
  warpgroup_wait<0>();asm volatile("":::"memory");
  #pragma unroll
  for(int j=0;j<4;++j)warpgroup_fence_operand(scores[j]);
SEED
  exponentiate(scores[0],_0{},cute::false_type{});
  pack(scores[0],pp);issue_pv(0,_0{},pp);
  exponentiate(scores[1],_1{},cute::false_type{});
  pack(scores[1],pp1);issue_pv(1,_1{},pp1);
  copy(scores[2],scores[0]);copy(scores[3],scores[1]);
  auto step=[&](auto seqc) __attribute__((always_inline)) {
    constexpr int seq=decltype(seqc)::value;
    auto& sc=scores[seq%2];auto& prob=seq%2==0?pp:pp1;
    // Entry: normally QK(k), PV(k-1), QK(k+1). wait1 retires the
    // current score and old P users, leaving QK(k+1) to overlap E(k).
    // Bootstrap has PV0/PV1; final body has QK95/PV94. wait1 is safe
    // for both, with a final drain before output and last V release.
    warpgroup_wait<1>();asm volatile("":::"memory");
    warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
    release_k(seq);release(seq-2);
    exponentiate(sc,Int<seq%H>{},cute::false_type{});
    pack(sc,prob);issue_pv(seq,Int<seq%H>{},prob);
    if constexpr(seq+2<768/N*H){
      init_score(sc,seq+2);issue_qk(sc,seq+2,Int<(seq+2)%H>{});
    }
  };
STEPS
  drain();release(768/N*H-2);release(768/N*H-1);
 }
'''.replace('SEED',seed).replace('STEPS','\n'.join('  step(Int<%d>{});'%i for i in range(2,96)))
s=s[:a]+body+s[b:]
source=HERE/(name+'.cu');source.write_text(s)
cpp=HERE/(name+'.cpp');cpp.write_text((HERE/(old+'.cpp')).read_text().replace(old,name))
directory=HERE/('build_'+name);directory.mkdir(exist_ok=True)
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
load(name='triattn_sol_'+name,sources=[str(cpp),str(source)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','--keep','--keep-dir='+str(directory)],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',name,flush=True)
