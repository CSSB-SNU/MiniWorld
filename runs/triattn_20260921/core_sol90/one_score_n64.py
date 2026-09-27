"""M128 N64: one score, two P fragments, four bias half-slots."""
def transform(s):
 from three_probability import transform as p3
 from independent_rows import transform as rows
 s=rows(p3(s,period24=True))
 s=s.replace('Score next_score;','').replace('Score third_score;','')
 s=s.replace('warpgroup_fence_operand(next_score);','').replace('warpgroup_fence_operand(third_score);','')
 s=s.replace('auto pp2=make_fragment_like(pp);','').replace('warpgroup_fence_operand(pp2);','')
 s=s.replace('bias[8][64*N]','bias[4][64*N]').replace('bf[4]','bf[2]').replace('be[8]','be[4]')
 s=s.replace('st<8','st<4').replace('if(st<4)','if(st<2)')
 s=s.replace('seq%8','seq%4').replace('seq/8','seq/4').replace('seq>=8','seq>=4')
 a=s.index(' }else{\n  clear(pp2);');b=s.index('\n auto id=',a)
 calls='\n'.join('   step(seq+%d,_%d{},cute::%s_type{});'%(i,i%2,'true' if i<2 else 'false') for i in range(8))
 hot=''' }else{
  clear(pp);clear(pp1);
  init_score(score,0);issue_qk(score,0,_0{});drain();
  auto step=[&](int seq,auto half,auto seed_possible) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value;
   auto& prob=hh==0?pp:pp1;
   warpgroup_wait<1>();warpgroup_fence_operand(score);warpgroup_fence_operand(prob);
   if(seq>=2)release(seq-2);
   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(score,half,cute::true_type{});else exponentiate(score,half,cute::false_type{});
   }else exponentiate(score,half,cute::false_type{});
   pack(score,prob);init_score(score,seq+1);
   warpgroup_fence_operand(prob);issue_qk(score,seq+1,Int<1-hh>{});
   issue_pv_prefenced(seq,half,prob);
  };
  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=8){
'''+calls+'''
   drain();
  }
  drain();release(2*nk-1);
 }
'''
 return s[:a]+hot+s[b:]
