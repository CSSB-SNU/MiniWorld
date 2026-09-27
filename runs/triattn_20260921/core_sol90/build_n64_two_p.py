"""N64 Q-register pipeline with two score and two probability buffers."""
from pathlib import Path
import hashlib, os
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
installed = HERE.parent / 'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest() == '6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant = 'm128s2p2qrn64'
s = (HERE / 'm128s2p3qrn64.cu').read_text().replace('m128s2p3qrn64', variant)
s = s.replace('auto pp1=make_fragment_like(pp);auto pp2=make_fragment_like(pp);', 'auto pp1=make_fragment_like(pp);')
s = s.replace('warpgroup_fence_operand(pp2);', '')
a = s.index('  clear(pp);clear(pp1);clear(pp2);')
b = s.index('\n auto id=', a)
s = s[:a] + '''  clear(pp);clear(pp1);
  init_score(score,0);issue_qk(score,0,_0{});init_score(next_score,1);drain();
  auto step=[&](int seq,auto half,auto seed_possible) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value;
   auto& sc=hh==0?score:next_score;auto& fs=hh==0?next_score:score;
   auto& prob=hh==0?pp:pp1;auto& prev=hh==0?pp1:pp;
   // Retire QK(k) and PV(k-3); newest PV(k-2) may still own prob.
   warpgroup_wait<1>();warpgroup_fence_operand(sc);
   warpgroup_fence_operand(prev);issue_qk(fs,seq+1,Int<1-hh>{});
   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   // Retire PV(k-2) before overwriting its register-A buffer. QK(k+1)
   // remains in flight, overlapping this pack and the next PV issue.
   warpgroup_wait<1>();warpgroup_fence_operand(prob);
   if(seq>=2)release(seq-2);
   pack(sc,prob);
   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prev);
   init_score(sc,seq+2);
  };
  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=8){
   step(seq+0,_0{},cute::true_type{});
   step(seq+1,_1{},cute::true_type{});
   step(seq+2,_0{},cute::false_type{});
   step(seq+3,_1{},cute::false_type{});
   step(seq+4,_0{},cute::false_type{});
   step(seq+5,_1{},cute::false_type{});
   step(seq+6,_0{},cute::false_type{});
   step(seq+7,_1{},cute::false_type{});
   drain();
  }
  issue_pv(2*nk-1,_1{},pp1);drain();release(2*nk-1);
 }
''' + s[b:]
assert 'pp2' not in s
src = HERE / (variant + '.cu')
src.write_text(s)
cpp = HERE / (variant + '.cpp')
cpp.write_text((HERE / 'm128s2p3qrn64.cpp').read_text().replace('m128s2p3qrn64', variant))
directory = HERE / ('build_' + variant)
directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
load(name='triattn_sol_' + variant, sources=[str(cpp), str(src)],
     extra_include_paths=[str(HERE.parents[1] / 'anthropic_adoption_20260919/cutlass-4.2/include')],
     extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda', '--use_fast_math', '-lineinfo', '-Xptxas=-v'],
     extra_ldflags=['-lcuda'], build_directory=str(directory), verbose=True)
print('BUILT', variant, flush=True)
