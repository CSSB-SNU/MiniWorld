"""R4/M128,544 threads, static register pool, two S/two P, compact N40 ones."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def transform():
 s=(HERE/'persistent_kv_body.cuh').read_text()
 s=s.replace('Rows=2,Stages=6','Rows=4,Stages=2')
 s=s.replace('Consumers=256,Threads=384','Consumers=512,Threads=544')
 s=s.replace('GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_32>>','GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_32>>')
 old='using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));'
 assert old in s
 s=s.replace(old,'''using PVBase=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));''')
 a=s.index('struct Shared {');b=s.index('__device__ __forceinline__ float ex2',a)
 s=s[:a]+'''struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D];
 alignas(1024) Element ones[N*D];
 alignas(128) float bias[8][64*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages],bf[8];
 cutlass::arch::ClusterBarrier empty[Stages],be[8];
};
static_assert(sizeof(Shared)<=232448,"R4 M128 shared storage exceeds H100");
'''+s[b:]
 s=s.replace('int tid=threadIdx.x,rg=tile%(L/Rows),h=tile/(L/Rows),i0=rg*Rows;',
             'int tid=threadIdx.x,qt=tile%6,rg=(tile/6)%(L/Rows),h=tile/(6*(L/Rows)),i0=rg*Rows;')
 a=s.index(' if(tid==0){');b=s.index(' for(int x=tid;',a)
 s=s[:a]+''' if(tid==0){
  s.qr.init(1);
  for(int st=0;st<Stages;++st){s.full[st].init(1);s.empty[st].init(Rows*4);}
  for(int st=0;st<8;++st){s.bf[st].init(1);s.be[st].init(Rows*4);}
  cutlass::arch::fence_barrier_init();
 }
'''+s[b:]
 s=s.replace('x<8*N','x<N*D')
 a=s.index(' if(tid>=Consumers){');b=s.index(' int wg=',a)
 s=s[:a]+(HERE/'r4_warp_producer.cuh').read_text()+s[b:]
 s=s.replace('QK qk;PV pv;LS ls;','QK qk;PV pv;PVBase pvbase;LS ls;')
 s=s.replace('auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);','auto tq=qk.get_slice(0);auto tp=pvbase.get_slice(0);')
 s=s.replace(' auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));\n','')
 s=s.replace('using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));','using Output=decltype(partition_fragment_C(pv,Shape<_64,_40>{}));')
 a=s.index(' auto qshared=');b=s.index('  Score sc[4];',a)
 s=s[:a]+' {\n'+s[b:]
 s=s.replace('Score sc[4];Output acc[2];Den den[2];','''Score sc[2];Output acc[2];
  auto d0=make_tensor(acc[0].data()+16,Den{}.layout());
  decltype(d0) den[2]={d0,make_tensor(acc[1].data()+16,Den{}.layout())};''')
 s=s.replace('s.qr.wait(qt&1)','s.qr.wait(0)')
 a=s.index('  #pragma unroll\n  for(int hh=0;hh<2;++hh){\n   auto src=qthr');b=s.index('  warpgroup_fence_operand(acc[0]);',a)
 s=s[:a]+s[b:]
 s=s.replace('s.bf[seq&1].wait((seq/2)&1)','s.bf[seq&7].wait((seq/8)&1)')
 s=s.replace('s.bias[seq&1]','s.bias[seq&7]').replace('s.be[seq&1]','s.be[seq&7]')
 s=s.replace('s.full[kt].wait(0)','s.full[kt%Stages].wait((kt/Stages)&1)')
 s=s.replace('s.k[kt<6?kt:0]','s.k[(kt<6?kt:0)%Stages]')
 s=s.replace('auto kb=tq.partition_fragment_B(ks);auto& qa=hh==0?qa0:qa1;',
 '''auto kb=tq.partition_fragment_B(ks);
   auto qs=local_tile(make_tensor(make_smem_ptr(s.q[wg]),SQ{}),Shape<_64,_32>{},make_coord(hh,0));
   auto qa=tq.partition_fragment_A(qs);''')
 s=s.replace('   if(seq==1){__syncwarp();if(t%32==0)s.qe.arrive();}\n','')
 s=s.replace('s.v[seq/8]','s.v[(seq/8)%Stages]')
 old='auto vb=tp.partition_fragment_B(vs);'
 assert s.count(old)==1
 s=s.replace(old,'''auto raw=tp.partition_fragment_B(vs);auto desc=raw.data().desc_;
   uint32_t va=cast_smem_ptr_to_uint(s.v[(seq/8)%Stages][wg])+uint32_t((seq/2)%4)*N*D*sizeof(Element);
   uint32_t oa=cast_smem_ptr_to_uint(s.ones);
   desc.bitfield.leading_byte_offset_=(oa-va)>>4;
   auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());''')
 s=s.replace('   #pragma unroll\n   for(int kk=0;kk<2;++kk)gemm(ls,prob(_,_,kk),lb(_,_,kk),den[hh]);\n','')
 s=s.replace('for(int n=0;n<4;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<2;++n)warpgroup_fence_operand(sc[n]);')
 mark='  if constexpr(Safe){\n   auto step='
 assert mark in s
 s=s.replace(mark,'''  auto release=[&](int seq) __attribute__((always_inline)){
   if(seq%8==7){__syncwarp();if(t%32==0)s.empty[(seq/8)%Stages].arrive();}
  };
'''+mark)
 old='exponentiate(sc[0],hc,cute::true_type{});pack(sc[0],pp[0]);issue_pv(pp[0],seq,hc,cute::false_type{});drain();'
 assert old in s
 s=s.replace(old,old+'release(seq);')
 a=s.index('  }else{\n   init(sc[0],0);');b=s.index('\n  auto id=',a)
 calls='\n'.join('    step(period*16+%d,Int<%d>{},cute::%s_type{});'%(i,i&1,'true' if i<2 else 'false') for i in range(16))
 s=s[:a]+'''  }else{
   init(sc[0],0);issue_qk(sc[0],0,_0{});drain();
   auto step=[&](int seq,auto hc,auto seed_possible) __attribute__((always_inline)){
    constexpr int hh=decltype(hc)::value;
    auto& cur=sc[hh];auto& future=sc[1-hh];
    auto& prob=pp[hh];auto& previous=pp[1-hh];
    // Retire QK(k), leaving PV(k-2). P(k-1) was written in the prior step.
    warpgroup_wait<1>();warpgroup_fence_operand(cur);
    init(future,seq+1);warpgroup_fence_operand(previous);
    issue_qk(future,seq+1,Int<1-hh>{});
    if constexpr(decltype(seed_possible)::value){
     if(seq<2)exponentiate(cur,hc,cute::true_type{});else exponentiate(cur,hc,cute::false_type{});
    }else exponentiate(cur,hc,cute::false_type{});
    // QK(k+1) is newest. This wait retires PV(k-2), freeing P(k)'s slot.
    warpgroup_wait<1>();warpgroup_fence_operand(prob);
    if(seq>=2)release(seq-2);
    pack(cur,prob);
    issue_pv(previous,max(seq-1,0),Int<1-hh>{},cute::true_type{});
   };
   #pragma unroll 1
   for(int period=0;period<3;++period){
'''+calls+'''
    drain();rescale(sc[1]);
    // Last odd E is still in S1. Repack after the exact rescale, before its PV.
    pack(sc[1],pp[1]);
   }
   issue_pv(pp[1],47,_1{},cute::false_type{});drain();release(47);
  }
'''+s[b:]
 s=s.replace('make_identity_tensor(Shape<_64,_32>{})','make_identity_tensor(Shape<_64,_40>{})')
 s=s.replace('for(int n=0;n<size(acc[hh]);n+=2)','for(int n=0;n<16;n+=2)')
 s=s.replace('  warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);\n','')
 includes=(HERE/'m128_three.cu').read_text().split('namespace SOL_NAMESPACE')[0]
 tail=(HERE/'m128_three.cu').read_text().split('__global__ void prepare(',1)[1]
 tail='__global__ void prepare('+tail
 tail=tail.replace('if(threadIdx.x==0){*valid=yes;*fix=0;}','if(threadIdx.x==0)*valid=yes;for(int x=threadIdx.x;x<6*(L/Rows)*4;x+=256)fix[x]=0;')
 tail=tail.replace('int L=q.size(-2);','int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale only");')
 return includes+s+tail
