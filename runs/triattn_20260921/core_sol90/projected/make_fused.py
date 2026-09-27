"""Construct initial fused producer/consumer kernel after projection proof.

Only generates a source body. Host integration and qualifications are separate.
"""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def generate(projection_source, producer_regs=64, consumer_regs=216):
    assert 128*(producer_regs+2*consumer_regs)<=384*168
    s=(ROOT/'persistent_kv_body.cuh').read_text()
    s=s.replace('namespace SOL_NAMESPACE','namespace triattn_projected_fused')
    s=s.replace('Stages=6','Stages=2')
    # The original QK register layout and arithmetic remain unchanged.
    start=s.index('struct Params {');end=s.index('__device__ __forceinline__ float ex2',start)
    p=projection_source
    aliases=p[p.index('using SX='):p.index('struct Params {')]
    aliases=aliases[:aliases.index('using TO=')]+aliases[aliases.index('using MQ='):]
    # Aliases SO/GO/DO are harmless; TX uses the normalized X orientation.
    s=s[:start]+aliases+'''struct Params {TX x;TW w;TWKV wkv;TB bias;Element* out;int* fix;int* valid;int L;float scale;};
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];
 alignas(128) float bias[8][64*N];
 alignas(128) Element x[Rows][128*128],w[64*128];
 cutlass::arch::ClusterBarrier qr,full[Stages],empty[Stages],be[8];
 cutlass::arch::ClusterTransactionBarrier xr[Rows],wr,bf[8];
};
static_assert(sizeof(Shared)<=232448,"projected attention exceeds H100 shared memory");
'''+s[end:]
    a=p.index('template<class MMA,class WTensor>');b=p.index('\n__global__',a)
    project=p[a:b].replace('int r,int hh,int tid','int r,int hh,int st,int tid')
    project=project.replace('s.x)', 's.x[r])').replace('s.k[0][r]','s.k[st][r]').replace('s.v[0][r]','s.v[st][r]')
    project=project.replace('Stages*R','Stages*Rows')
    a=s.index('template<bool Safe>')
    s=s[:a]+project+'\n'+s[a:]
    s=s.replace('int tid=threadIdx.x,rg=tile%(L/Rows),h=tile/(L/Rows),i0=rg*Rows;',
                'int tid=threadIdx.x,qt=tile%6,rg=(tile/6)%(L/Rows),h=tile/(6*(L/Rows)),i0=rg*Rows;')
    a=s.index(' if(tid==0){');b=s.index(' for(int x=tid;',a)
    s=s[:a]+''' if(tid==0){
  s.qr.init(1);s.wr.init(1);
  for(int r=0;r<Rows;++r)s.xr[r].init(1);
  for(int st=0;st<Stages;++st){s.full[st].init(1);s.empty[st].init(Rows*4);}
  for(int st=0;st<8;++st){s.bf[st].init(1);s.be[st].init(Rows*4);}
  cutlass::arch::fence_barrier_init();
 }
'''+s[b:]
    a=s.index(' if(tid>=Consumers){');b=s.index(' cutlass::arch::warpgroup_reg_alloc<224>();',a)
    producer=(HERE/'fused_producer.cuh').read_text().replace('PRODUCER_REGS',str(producer_regs))
    s=s[:a]+producer+s[b:]
    s=s.replace('warpgroup_reg_alloc<224>','warpgroup_reg_alloc<%d>'%consumer_regs)
    s=s.replace(' #pragma unroll 1\n for(int qt=0;qt<6;++qt){',' {')
    s=s.replace('s.qr.wait(qt&1)','s.qr.wait(0)')
    s=s.replace('s.bf[seq&1].wait((seq/2)&1)','s.bf[seq&7].wait((seq/8)&1)')
    s=s.replace('s.bias[seq&1]','s.bias[seq&7]').replace('s.be[seq&1]','s.be[seq&7]')
    s=s.replace('s.full[kt].wait(0)','s.full[kt%Stages].wait((kt/Stages)&1)')
    s=s.replace('s.k[kt<6?kt:0]','s.k[(kt<6?kt:0)%Stages]')
    s=s.replace('s.v[seq/8]','s.v[(seq/8)%Stages]')
    s=s.replace('   if(seq==1){__syncwarp();if(t%32==0)s.qe.arrive();}\n','')
    mark='  if constexpr(Safe){\n   auto step='
    assert mark in s
    s=s.replace(mark,'''  auto release=[&](int kt) __attribute__((always_inline)){
   __syncwarp();if(t%32==0)s.empty[kt%Stages].arrive();
  };
'''+mark)
    mark='exponentiate(sc[0],hc,cute::true_type{});pack(sc[0],pp[0]);issue_pv(pp[0],seq,hc,cute::false_type{});drain();'
    assert mark in s
    s=s.replace(mark,mark+'\n    if(seq%8==7)release(seq/8);')
    mark='    if constexpr(dd!=0)warpgroup_wait<2>();'
    assert mark in s
    # dd0 follows a full drain at each period boundary. PV(seq-3) has retired
    # in both branches; release a KV stage only after its last PV (seq-3).
    s=s.replace(mark,mark+'\n    if(seq>=10 && seq%8==2)release((seq-3)/8);')
    includes=(ROOT/'m128_three.cu').read_text().split('namespace SOL_NAMESPACE')[0]
    includes=includes.replace('../core_tiles/r1/csrc/fa3_utils.h','../../core_tiles/r1/csrc/fa3_utils.h')
    return includes+s

if __name__=='__main__':
    source=(HERE/'projection_stsm.cu').read_text()
    (HERE/'fused_body.cuh').write_text(generate(source))
