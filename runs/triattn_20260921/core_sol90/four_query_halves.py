"""M256 / two-score / two-P experiment with four output accumulators."""
def transform(s):
 from make_wide import transform as denominator
 from independent_rows import transform as rows
 s=rows(denominator(s))
 s=s.replace('M=128','M=256').replace('Shape<_128,_32>','Shape<_256,_32>')
 s=s.replace('2*Ratio','4*Ratio').replace('(seq/2)','(seq/4)')
 s=s.replace('2*nk','4*nk').replace('2*768/N','4*768/N').replace('2*L/N','4*L/N')
 s=s.replace('nk=768/N','nk=p.L/N')
 s=s.replace('Score third_score;','').replace('warpgroup_fence_operand(third_score);','')
 s=s.replace('Output acc[2];clear(acc[0]);clear(acc[1]);',
             'Output acc[4];clear(acc[0]);clear(acc[1]);clear(acc[2]);clear(acc[3]);')
 s=s.replace('Den den[2];clear(den[0]);clear(den[1]);',
             'Den den[4];clear(den[0]);clear(den[1]);clear(den[2]);clear(den[3]);')
 s=s.replace('warpgroup_fence_operand(acc[1]);',
             'warpgroup_fence_operand(acc[1]);warpgroup_fence_operand(acc[2]);warpgroup_fence_operand(acc[3]);')
 s=s.replace('warpgroup_fence_operand(den[1]);',
             'warpgroup_fence_operand(den[1]);warpgroup_fence_operand(den[2]);warpgroup_fence_operand(den[3]);')
 s=s.replace('float mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};',
             'float mx[4][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};')
 s=s.replace('for(int hh=0;hh<2;hh++)','for(int hh=0;hh<4;hh++)')
 s=s.replace('hh=(idx/(64*N))%2','hh=(idx/(64*N))%4')
 a=s.index(' if constexpr(Safe){\n  auto step=');b=s.index('\n auto id=',a)
 calls='\n'.join('   step(seq+%d,_%d{},cute::%s_type{});'%(i,i%4,'true' if i<4 else 'false') for i in range(32))
 s=s[:a]+''' if constexpr(Safe){
  auto step=[&](int seq,auto half) __attribute__((always_inline)) {
   init_score(score,seq);issue_qk(score,seq,half);warpgroup_wait<0>();warpgroup_fence_operand(score);
   exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);drain();release(seq);
  };
  for(int seq=0;seq<4*nk;seq+=4){step(seq,_0{});step(seq+1,_1{});step(seq+2,_2{});step(seq+3,_3{});}
 }else{
  init_score(score,0);issue_qk(score,0,_0{});drain();
  auto step=[&](int seq,auto half,auto seed_possible) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value;
   auto& sc=(hh%2)==0?score:next_score;
   auto& ns=(hh%2)==0?next_score:score;
   auto& prob=(hh%2)==0?pp:pp1;
   init_score(ns,seq+1);
   warpgroup_wait<1>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   if(seq>=2)release(seq-2);
   issue_qk(ns,seq+1,Int<(hh+1)%4>{});
   if constexpr(decltype(seed_possible)::value){
    if(seq<4)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   pack(sc,prob);issue_pv(seq,half,prob);
  };
  #pragma unroll 1
  for(int seq=0;seq<4*nk;seq+=32){
'''+calls+'''
   drain();
  }
  drain();release(4*nk-1);
 }
'''+s[b:]
 return s
