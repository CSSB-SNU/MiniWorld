"""Three scores and four P fragments: QK three chunks ahead, wait4."""
def transform(s):
 from three_probability import transform as p3
 from independent_rows import transform as rows
 s=rows(p3(s,period24=True),split=True)
 s=s.replace('auto pp2=make_fragment_like(pp);','auto pp2=make_fragment_like(pp);auto pp3=make_fragment_like(pp);')
 s=s.replace('warpgroup_fence_operand(pp2);','warpgroup_fence_operand(pp2);warpgroup_fence_operand(pp3);')
 a=s.index(' }else{\n  clear(pp2);');b=s.index('\n auto id=',a)
 calls='\n'.join('   step(seq+%d,_%d{},_%d{},_%d{},cute::%s_type{});'%(i,i%2,i%3,i%4,'true' if i<2 else 'false') for i in range(24))
 hot=''' }else{
  clear(pp3);
  init_score(score,0);issue_qk(score,0,_0{});
  init_score(next_score,1);issue_qk(next_score,1,_1{});
  init_score(third_score,2);issue_qk(third_score,2,_0{});drain();
  // Six committed groups at steady state. wait4 retires QK(k) and
  // PV(k-4), freeing score k%3 and probability k%4. QK(k+3) reuses
  // the score only after E(k) has been packed; PV(k-1) commits last.
  auto step=[&](int seq,auto half,auto which,auto slot,auto seed_possible) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value,ci=decltype(which)::value;
   constexpr int pi=decltype(slot)::value,prev=(pi+3)%4;
   auto& sc=ci==0?score:(ci==1?next_score:third_score);
   auto& prob=pi==0?pp:(pi==1?pp1:(pi==2?pp2:pp3));
   auto& prior=prev==0?pp:(prev==1?pp1:(prev==2?pp2:pp3));
   warpgroup_wait<4>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   release_k(seq);if(seq>=4)release(seq-4);
   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   pack(sc,prob);init_score(sc,seq+3);
   warpgroup_fence_operand(prior);issue_qk(sc,seq+3,Int<1-hh>{});
   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prior);
  };
  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=24){
'''+calls+'''
   drain();
  }
  issue_pv(2*nk-1,_1{},pp3);drain();release(2*nk-1);
 }
'''
 return s[:a]+hot+s[b:]
