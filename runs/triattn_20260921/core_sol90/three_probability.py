"""Three score / three packed-P buffers keep delayed PV with fewer registers."""
def transform(s,period24=False):
 from make_wide import transform as denominator
 s=denominator(s)
 s=s.replace(' auto pp1=make_fragment_like(pp);',' auto pp1=make_fragment_like(pp);auto pp2=make_fragment_like(pp);')
 s=s.replace('warpgroup_fence_operand(pp1);','warpgroup_fence_operand(pp1);warpgroup_fence_operand(pp2);')
 a=s.index(' auto issue_pv=');b=s.index(' auto release=',a)
 helper=s[a:b].replace('auto issue_pv=', 'auto issue_pv_prefenced=')
 helper=helper.replace('warpgroup_fence_operand(pp);','').replace('  warpgroup_arrive();\n','')
 s=s[:b]+helper+s[b:]
 a=s.index(' }else{\n  init_score(score,0);');b=s.index('\n auto id=',a)
 hot=''' }else{
  init_score(score,0);init_score(next_score,1);
  issue_qk(score,0,_0{});issue_qk(next_score,1,_1{});
  warpgroup_wait<0>();warpgroup_fence_operand(score);warpgroup_fence_operand(next_score);
  exponentiate(score,_0{},cute::true_type{});pack(score,pp);issue_pv(0,_0{},pp);
  init_score(third_score,2);issue_qk(third_score,2,_0{});
  exponentiate(next_score,_1{},cute::true_type{});pack(next_score,pp1);
  init_score(score,3);issue_qk(score,3,_1{});init_score(next_score,4);drain();
  // At k: wait retires QK(k) and PV(k-3). The freed P slot receives
  // packed E(k), while already-packed P(k-1) is issued after E without
  // a second hardware fence. Three score and three P buffers suffice.
  auto step=[&](int seq,auto half,auto which) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value,ci=decltype(which)::value,fi=(ci+2)%3;
   auto& sc=ci==0?score:(ci==1?next_score:third_score);
   auto& fs=fi==0?score:(fi==1?next_score:third_score);
   auto& prob=ci==0?pp:(ci==1?pp1:pp2);
   auto& prev=fi==0?pp:(fi==1?pp1:pp2);
   warpgroup_wait<2>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   if(seq>=3)release(seq-3);
   warpgroup_fence_operand(prev);issue_qk(fs,seq+2,half);
   exponentiate(sc,half,cute::false_type{});pack(sc,prob);
   issue_pv_prefenced(seq-1,Int<1-hh>{},prev);
   init_score(sc,seq+3);
  };
  int seq=2;
  #pragma unroll 1
  for(;seq+5<2*nk;seq+=6){
   step(seq,_0{},_2{});step(seq+1,_1{},_0{});step(seq+2,_0{},_1{});
   step(seq+3,_1{},_2{});step(seq+4,_0{},_0{});step(seq+5,_1{},_1{});drain();
  }
  if(seq<2*nk){step(seq,_0{},_2{});step(seq+1,_1{},_0{});drain();}
  if(seq+2<2*nk){step(seq+2,_0{},_1{});step(seq+3,_1{},_2{});drain();}
  drain();
  static_assert((2*(768/N)-1)%3==2);
  issue_pv(2*nk-1,_1{},pp2);drain();release(2*nk-1);
 }
'''
 s=s[:a]+hot+s[b:]
 if period24:
  # Twenty-four chunks align the score/P rings (3), query halves (2),
  # and eight-chunk K/V and bias tiles. A runtime bound preserves the
  # loop; the host contract still restricts this variant to L768.
  s=s.replace('nk=768/N','nk=p.L/N')
  a=s.index(' }else{\n  init_score(score,0);');b=s.index('\n auto id=',a)
  body=hot[hot.index('  auto step='):hot.index('  int seq=2;')]
  body=body.replace('auto which)', 'auto which,auto seed_possible)')
  body=body.replace('   exponentiate(sc,half,cute::false_type{});pack(sc,prob);', '''   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   pack(sc,prob);''')
  body=body.replace('issue_pv_prefenced(seq-1,','issue_pv_prefenced(max(seq-1,0),')
  calls='\n'.join('   step(seq+%d,_%d{},_%d{},cute::%s_type{});'%(i,i%2,i%3,'true' if i<2 else 'false') for i in range(24))
  new=''' }else{
  clear(pp2);
  init_score(score,0);issue_qk(score,0,_0{});
  init_score(next_score,1);issue_qk(next_score,1,_1{});
  init_score(third_score,2);drain();
'''+body+'''
  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=24){
'''+calls+'''
   drain();
  }
  issue_pv(2*nk-1,_1{},pp2);drain();release(2*nk-1);
 }
'''
  s=s[:a]+new+s[b:]
 return s
