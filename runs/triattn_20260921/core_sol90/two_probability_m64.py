"""M64 two-score/two-P pipeline, prefetched bias and one wait per chunk."""
def transform(s):
 from four_probability import transform as base
 from m64_probability import transform as m64
 s=m64(base(s))
 s=s.replace('Score third_score;','').replace('warpgroup_fence_operand(third_score);','')
 for name in ['pp2','pp3']:
  s=s.replace('auto '+name+'=make_fragment_like(pp);','')
  s=s.replace('warpgroup_fence_operand('+name+');','')
 a=s.index(' }else{\n  clear(pp3);');b=s.index('\n auto id=',a)
 calls='\n'.join('   step(seq+%d,_%d{},cute::%s_type{});'%(i,i%2,'true' if i==0 else 'false') for i in range(8))
 hot=''' }else{
  clear(pp);clear(pp1);
  init_score(score,0);issue_qk(score,0,_0{});
  init_score(next_score,1);drain();
  auto step=[&](int seq,auto which,auto seed_possible) __attribute__((always_inline)) {
   constexpr int ci=decltype(which)::value;
   auto& sc=ci==0?score:next_score;auto& ns=ci==0?next_score:score;
   auto& prob=ci==0?pp:pp1;
   warpgroup_wait<1>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   release_k(seq);if(seq>=2)release(seq-2);
   issue_qk(ns,seq+1,_0{});
   if constexpr(decltype(seed_possible)::value){
    if(seq==0)exponentiate(sc,_0{},cute::true_type{});else exponentiate(sc,_0{},cute::false_type{});
   }else exponentiate(sc,_0{},cute::false_type{});
   pack(sc,prob);issue_pv(seq,_0{},prob);
   init_score(sc,seq+2);
  };
  #pragma unroll 1
  for(int seq=0;seq<nk;seq+=8){
'''+calls+'''
   drain();
  }
  drain();release(nk-1);
 }
'''
 return s[:a]+hot+s[b:]
