 }else{
  init_score(score,0);init_score(next_score,1);
  issue_qk(score,0,_0{});issue_qk(next_score,1,_1{});
  warpgroup_wait<0>();warpgroup_fence_operand(score);warpgroup_fence_operand(next_score);
  exponentiate(score,_0{},cute::true_type{});pack(score,pp);issue_pv(0,_0{},pp);
  init_score(third_score,2);issue_qk(third_score,2,_0{});
  exponentiate(next_score,_1{},cute::true_type{});pack(next_score,pp1);issue_pv(1,_1{},pp1);
  init_score(score,3);issue_qk(score,3,_1{});init_score(next_score,4);
  auto step=[&](int seq,auto half,auto which) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value,ci=decltype(which)::value,fi=(ci+2)%3;
   auto& sc=ci==0?score:(ci==1?next_score:third_score);
   auto& fs=fi==0?score:(fi==1?next_score:third_score);
   auto& prob=hh==0?pp:pp1;
   warpgroup_wait<2>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   release(seq-2);issue_qk(fs,seq+2,half);
   exponentiate(sc,half,cute::false_type{});pack(sc,prob);issue_pv(seq,half,prob);
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
  drain();release(2*nk-1);
 }
