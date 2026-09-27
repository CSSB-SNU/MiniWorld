"""M128/R2/N32, two independent producer warps, one S/one P, two CTAs/SM."""
from pathlib import Path
from r4_warp import transform as base
HERE=Path(__file__).resolve().parent

def transform(period_size=16,pv_fence=False):
 assert period_size in (6,16)
 s=base().replace('Rows=4,Stages=2','Rows=2,Stages=2')
 s=s.replace('Consumers=512,Threads=544','Consumers=256,Threads=320')
 s=s.replace('__launch_bounds__(Threads,1)','__launch_bounds__(Threads,2)')
 s=s.replace('bias[8][64*N]','bias[3][64*N]').replace('bf[8]','bf[3]').replace('be[8]','be[3]')
 s=s.replace('for(int st=0;st<8;++st)','for(int st=0;st<3;++st)')
 a=s.index('if(tid>=Consumers){');b=s.index(' int wg=',a)
 s=s[:a]+''' if(tid>=Consumers){
  // No WGMMA or dynamic register reconfiguration in these two producer warps.
  if(tid==Consumers){
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   #pragma unroll
   for(int r=0;r<Rows;++r){
    auto src=local_tile(p.q.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   #pragma unroll 1
   for(int kt=0;kt<6;++kt){
    int st=kt%Stages;
    if(kt>=Stages){s.empty[st].wait(((kt/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element));
    #pragma unroll
    for(int r=0;r<Rows;++r){
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
     auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }
   }
  }
  if(tid==Consumers+32){
   #pragma unroll 1
   for(int seq=0;seq<48;++seq){
    int st=seq%3;
    if(seq>=3){s.be[st].wait(((seq/3)-1)&1);asm volatile("":::"memory");}
    auto src=p.bias.get_tma_tensor(make_shape(256,8,48,6,4))(_,_,seq,qt,h);
    auto dst=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto sl=p.bias.get_slice(_0{});
    s.bf[st].arrive_and_expect_tx(64*N*sizeof(float));
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),sl.partition_S(src),sl.partition_D(dst));
   }
  }
  return;
 }
'''+s[b:]
 s=s.replace('s.bf[seq&7].wait((seq/8)&1)','s.bf[seq%3].wait((seq/3)&1)')
 s=s.replace('s.bias[seq&7]','s.bias[seq%3]').replace('s.be[seq&7]','s.be[seq%3]')
 s=s.replace('Score sc[2];','Score sc[1];').replace('decltype(pproto) pp[2];','decltype(pproto) pp[1];')
 s=s.replace('clear(pp[0]);clear(pp[1]);','clear(pp[0]);')
 s=s.replace('for(int n=0;n<2;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<1;++n)warpgroup_fence_operand(sc[n]);')
 s=s.replace('warpgroup_fence_operand(pp[hh]);','')
 s=s.replace('   warpgroup_wait<0>();','   warpgroup_wait<0>();warpgroup_fence_operand(pp[0]);')
 a=s.index('      if(hh==1){');b=s.index('\n     }\n    }',a)
 s=s[:a]+s[b:]
 a=s.index('  }else{\n   init(sc[0],0);');b=s.index('\n  auto id=',a)
 calls='\n'.join('    step(period*%d+%d,Int<%d>{},cute::%s_type{});'%(period_size,i,i&1,'true' if i<2 else 'false') for i in range(period_size))
 s=s[:a]+'''  }else{
   init(sc[0],0);issue_qk(sc[0],0,_0{});drain();
   auto step=[&](int seq,auto hc,auto seed_possible) __attribute__((always_inline)){
    constexpr int hh=decltype(hc)::value;
    // QK(k) precedes PV(k-1). E(k) can overlap the latter with one S slot.
    warpgroup_wait<1>();warpgroup_fence_operand(sc[0]);
    if constexpr(decltype(seed_possible)::value){
     if(seq<2)exponentiate(sc[0],hc,cute::true_type{});else exponentiate(sc[0],hc,cute::false_type{});
    }else exponentiate(sc[0],hc,cute::false_type{});
    warpgroup_wait<0>();warpgroup_fence_operand(pp[0]);
    if(seq>=1)release(seq-1);
    pack(sc[0],pp[0]);init(sc[0],seq+1);warpgroup_fence_operand(pp[0]);
    issue_qk(sc[0],seq+1,Int<1-hh>{});
    issue_pv(pp[0],seq,hc,cute::true_type{});
   };
   #pragma unroll 1
   for(int period=0;period<3;++period){
'''+calls+'''
    drain();rescale(sc[0]);
   }
   release(47);
  }
 '''+s[b:]
 if period_size!=16:
  s=s.replace('for(int period=0;period<3;++period)','for(int period=0;period<8;++period)')
 if pv_fence:
  old='if constexpr(!decltype(prefenced)::value){warpgroup_fence_operand(prob);warpgroup_arrive();}'
  assert old in s
  s=s.replace(old,'warpgroup_fence_operand(prob);warpgroup_arrive();')
 marker=' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);'
 assert marker in s
 s=s.replace(marker,''' int resident=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(resident>=2,"two-CTA resource target not met: ",resident," shared=",sizeof(Shared));
'''+marker)
 return s
