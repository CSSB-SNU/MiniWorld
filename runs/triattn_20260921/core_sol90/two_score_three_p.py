"""Two scores / three packed P buffers, QK one chunk ahead, PV one late."""
def transform(s):
 from three_probability import transform as three
 s=three(s)
 s=s.replace('Score third_score;','').replace('warpgroup_fence_operand(third_score);','')
 a=s.index(' }else{\n  init_score(score,0);');b=s.index('\n auto id=',a)
 steps='\n'.join('   step(seq+%d,_%d{},_%d{},cute::%s_type{});'%(i,i%2,i%3,'true' if i<2 else 'false') for i in range(24))
 hot=''' }else{
  clear(pp);clear(pp1);clear(pp2);
  init_score(score,0);issue_qk(score,0,_0{});init_score(next_score,1);drain();
  auto step=[&](int seq,auto half,auto which,auto seed_possible) __attribute__((always_inline)) {
   constexpr int si=decltype(half)::value,hh=decltype(half)::value,pi=decltype(which)::value,prev_i=(pi+2)%3;
   auto& sc=si==0?score:next_score;auto& fs=si==0?next_score:score;
   auto& prob=pi==0?pp:(pi==1?pp1:pp2);
   auto& prev=prev_i==0?pp:(prev_i==1?pp1:pp2);
   // The newest group is PV(k-2), which may still be in flight. QK(k)
   // and PV(k-3) are retired, so S(k) and the recycled P slot are ready.
   warpgroup_wait<1>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   if(seq>=3)release(seq-3);
   warpgroup_fence_operand(prev);issue_qk(fs,seq+1,Int<1-hh>{});
   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   pack(sc,prob);
   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prev);
   init_score(sc,seq+2);
  };
  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=24){
'''+steps+'''
   drain();
  }
  issue_pv(2*nk-1,_1{},pp2);drain();release(2*nk-1);
 }
'''
 s=s[:a]+hot+s[b:]
 assert 'third_score' not in s
 return s


def pack_late(s):
 old="   pack(sc,prob);\n   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prev);"
 assert s.count(old)==1
 return s.replace(old,"   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prev);\n   pack(sc,prob);")

def force_p_before_qk(s,sync=False):
 """Order every packed P register before QK through a real dependency."""
 s=s.replace('int L;float scale; };','int L;float scale;int zero; };')
 s=s.replace('valid.data_ptr<int>(),L,float(scale)};', 'valid.data_ptr<int>(),L,float(scale),0};')
 old='   warpgroup_fence_operand(prev);issue_qk(fs,seq+1,Int<1-hh>{});'
 assert s.count(old)==1
 dep='''   warpgroup_fence_operand(prev);
   auto packed_prev=recast<uint32_t>(prev);uint32_t dep=0;
   #pragma unroll
   for(int d=0;d<size(packed_prev);++d)dep|=packed_prev(d);
   uint32_t zero_dep=dep&uint32_t(p.zero);
'''
 if sync:
  dep+='   asm volatile("bar.warp.sync %0;"::"r"(zero_dep^0xffffffffu):"memory");\n'
 else:
  dep+='   asm volatile("add.rn.f32 %0,%0,%1;":"+f"(fs(0)):"f"(__uint_as_float(zero_dep)));\n'
 return s.replace(old,dep+'   issue_qk(fs,seq+1,Int<1-hh>{});')
