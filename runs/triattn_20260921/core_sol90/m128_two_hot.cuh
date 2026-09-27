 }else{
  issue_qk(score,0,_0{});warpgroup_wait<0>();warpgroup_fence_operand(score);
  issue_qk(next_score,1,_1{});exponentiate(score,_0{},cute::true_type{});pack(score);issue_pv(0,_0{});
  auto step=[&](int seq,auto half,auto seed) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value;
   auto& sc=hh==0?score:next_score;auto& ns=hh==0?next_score:score;
   warpgroup_wait<1>();warpgroup_fence_operand(sc);
   issue_qk(ns,seq+1,Int<1-hh>{});exponentiate(sc,half,seed);
   warpgroup_wait<1>();warpgroup_fence_operand(pp);release(seq-1);
   pack(sc);issue_pv(seq,half);
  };
  step(1,_1{},cute::true_type{});
  int seq=2;
  #pragma unroll 1
  for(;seq+7<2*nk;seq+=8){
   step(seq,_0{},cute::false_type{});step(seq+1,_1{},cute::false_type{});
   step(seq+2,_0{},cute::false_type{});step(seq+3,_1{},cute::false_type{});
   step(seq+4,_0{},cute::false_type{});step(seq+5,_1{},cute::false_type{});
   step(seq+6,_0{},cute::false_type{});step(seq+7,_1{},cute::false_type{});drain();
  }
  #pragma unroll 1
  for(;seq<2*nk;seq+=2){step(seq,_0{},cute::false_type{});step(seq+1,_1{},cute::false_type{});drain();}
  drain();release(2*nk-1);
 }
