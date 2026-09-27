"""Two-score/one-P pipeline; retire PV before repacking its sole register slot."""
from pathlib import Path
from r4_warp import transform as base
from r4_cooperative import transform as cooperate
HERE=Path(__file__).resolve().parent

def transform(rows=5):
 s=cooperate(base())
 s=s.replace('decltype(pproto) pp[2];','decltype(pproto) pp[1];')
 s=s.replace('clear(pp[0]);clear(pp[1]);','clear(pp[0]);')
 s=s.replace('warpgroup_fence_operand(pp[hh]);','')
 s=s.replace('   warpgroup_wait<0>();','   warpgroup_wait<0>();warpgroup_fence_operand(pp[0]);')
 # Every E is packed and submitted before the drain. There is no pending E
 # to rescale or repack, unlike the lagged-PV two-P pipeline.
 a=s.index('      if(hh==1){');b=s.index('\n     }\n    }',a)
 s=s[:a]+s[b:]
 a=s.index('  }else{\n   init(sc[0],0);');b=s.index('\n  auto id=',a)
 calls='\n'.join('    step(period*16+%d,Int<%d>{},cute::%s_type{});'%(i,i&1,'true' if i<2 else 'false') for i in range(16))
 s=s[:a]+'''  }else{
   init(sc[0],0);issue_qk(sc[0],0,_0{});drain();
   auto step=[&](int seq,auto hc,auto seed_possible) __attribute__((always_inline)){
    constexpr int hh=decltype(hc)::value;
    auto& cur=sc[hh];auto& future=sc[1-hh];auto& prob=pp[0];
    warpgroup_wait<1>();warpgroup_fence_operand(cur);
    if(tid==0 && seq+8<48)load_bias(seq+8);
    init(future,seq+1);issue_qk(future,seq+1,Int<1-hh>{});
    if constexpr(decltype(seed_possible)::value){
     if(seq<2)exponentiate(cur,hc,cute::true_type{});else exponentiate(cur,hc,cute::false_type{});
    }else exponentiate(cur,hc,cute::false_type{});
    // PV(seq-1) is older than QK(seq+1); release its operand only now.
    warpgroup_wait<1>();warpgroup_fence_operand(prob);
    if(seq>=1)release(seq-1);
    pack(cur,prob);issue_pv(prob,seq,hc,cute::false_type{});
   };
   #pragma unroll 1
   for(int period=0;period<3;++period){
'''+calls+'''
    drain();rescale(sc[1]);
   }
   release(47);
  }
'''+s[b:]
 if rows==5:
  s=s.replace('M=128,N=32,LN=128,Ratio=4,D=32,Rows=4,Stages=2','M=128,N=32,LN=64,Ratio=2,D=32,Rows=5,Stages=3')
  s=s.replace('Consumers=512,Threads=512','Consumers=640,Threads=640')
  s=s.replace('using SK=SQ;','using SK=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));')
  s=s.replace('Shape<_32,_128>{}','Shape<_32,_64>{}')
  s=s.replace('using TK=TQ;','using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SK{},Shape<_64,_32>{},_1{}));')
  s=s.replace('(L/Rows)','((L+Rows-1)/Rows)')
  s=s.replace('(i0+r)*4+h','min(i0+r,L-1)*4+h')
  s=s.replace('Shape<_128,_32>{},make_coord(kt,0)','Shape<_64,_32>{},make_coord(kt,0)')
  s=s.replace('load_kv(0);load_kv(1);','load_kv(0);load_kv(1);load_kv(2);')
  s=s.replace('int kt=seq/8;','int kt=seq/4;')
  s=s.replace('seq%8==0','seq%4==0')
  s=s.replace('kt<6?kt:0','kt<12?kt:0')
  s=s.replace('(seq/2)%4','(seq/2)%2')
  s=s.replace('(seq/8)%Stages','(seq/4)%Stages')
  s=s.replace('seq%8==7','seq%4==3')
  s=s.replace('seq/8+2<6','seq/4+3<12').replace('load_kv(seq/8+2)','load_kv(seq/4+3)')
  old='  auto id=pv.get_slice(t).partition_C'
  assert old in s
  s=s.replace(old,'  if(i0+wg<L){\n'+old)
  mark='  if constexpr(!Safe){if(__any_sync'
  assert mark in s
  s=s.replace(mark,'  }\n'+mark)
 elif rows!=4: raise ValueError(rows)
 return s
