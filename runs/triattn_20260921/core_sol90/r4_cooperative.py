"""Four full consumer WGs; thread0 also services TMA between MMA phases."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def transform(s):
 s=s.replace('Consumers=512,Threads=544','Consumers=512,Threads=512')
 a=s.index('if(tid>=Consumers){');b=s.index(' int wg=',a)
 producer=(HERE/'r4_warp_producer.cuh').read_text()
 qpart=producer[producer.index('  s.qr.arrive_and_expect_tx'):producer.index('  #pragma unroll 1\n  for(int kt=')]
 kvpart=producer[producer.index('   int st=kt%Stages;'):producer.index('   // Feed bias')]
 # The KV refill function waits for all four consumer WGs before reuse.
 helpers=''' auto load_kv=[&](int kt) __attribute__((always_inline)){
'''+kvpart+''' };
 auto load_bias=[&](int seq) __attribute__((always_inline)){
  int st=seq&7;
  if(seq>=8){s.be[st].wait(((seq/8)-1)&1);asm volatile("":::"memory");}
  auto src=p.bias.get_tma_tensor(make_shape(256,8,48,6,4))(_,_,seq,qt,h);
  auto dst=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto sl=p.bias.get_slice(_0{});
  s.bf[st].arrive_and_expect_tx(64*N*sizeof(float));
  copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),sl.partition_S(src),sl.partition_D(dst));
 };
 if(tid==0){
'''+qpart+'''
  load_kv(0);load_kv(1);
  #pragma unroll 1
  for(int seq=0;seq<8;++seq)load_bias(seq);
 }
'''
 s=s[:a]+helpers+s[b:]
 old='if(seq%8==7){__syncwarp();if(t%32==0)s.empty[(seq/8)%Stages].arrive();}'
 assert old in s
 s=s.replace(old,'''if(seq%8==7){
    __syncwarp();if(t%32==0)s.empty[(seq/8)%Stages].arrive();
    if(tid==0 && seq/8+2<6)load_kv(seq/8+2);
   }''')
 old='    warpgroup_wait<1>();warpgroup_fence_operand(cur);'
 assert old in s
 s=s.replace(old,old+'\n    if(tid==0 && seq+8<48)load_bias(seq+8);')
 old='drain();release(seq);'
 assert s.count(old)==1
 s=s.replace(old,old+'\n    if(tid==0 && seq+8<48)load_bias(seq+8);')
 return s
