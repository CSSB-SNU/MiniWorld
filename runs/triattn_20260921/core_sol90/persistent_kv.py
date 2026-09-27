"""Construct the persistent-K/V prototype with established prepare and uniform paths."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def transform(s):
 includes=s[:s.index('namespace SOL_NAMESPACE')]
 tail=s[s.index('__global__ void prepare('):]
 old='if(threadIdx.x==0){*valid=yes;*fix=0;}'
 assert old in tail
 tail=tail.replace(old,'if(threadIdx.x==0)*valid=yes;for(int x=threadIdx.x;x<(L/Rows)*4;x+=256)fix[x]=0;')
 tail=tail.replace('int L=q.size(-2);','int L=q.size(-2);TORCH_CHECK(L==768,"persistent K/V prototype requires L768");')
 tail=tail.replace('int ct=(L/M)*((L+Rows-1)/Rows)*4;','int ct=((L+Rows-1)/Rows)*4;')
 return includes+(HERE/'persistent_kv_body.cuh').read_text()+tail

def alias_q(s):
 """Reuse consumed shared Q as bias slots2/3, with a query-end rendezvous."""
 old=' alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];'
 assert old in s
 s=s.replace(old,''' union alignas(128) { Element q[Rows][M*D];float qb[2][64*N]; };
 alignas(128) Element k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];''')
 s=s.replace('full[Stages],bf[2]','full[Stages],bf[4]').replace('qe,be[2]','qe,qd,be[4]')
 s=s.replace('s.qr.init(1);s.qe.init(Rows*4);','s.qr.init(1);s.qe.init(Rows*4);s.qd.init(Rows*4);')
 s=s.replace('for(int st=0;st<2;++st){s.bf','for(int st=0;st<4;++st){s.bf')
 s=s.replace('if(qt){s.qe.wait((qt-1)&1);','if(qt){s.qd.wait((qt-1)&1);')
 s=s.replace('int st=g&1;', 'int st=g&3;\n    if(g%48==2){s.qe.wait((g/48)&1);asm volatile("":::"memory");}')
 s=s.replace('if(g>=2){s.be[st].wait(((g/2)-1)&1);','if(g>=4){s.be[st].wait(((g/4)-1)&1);')
 s=s.replace('make_smem_ptr(s.bias[st]),SB{}','make_smem_ptr(st<2?s.bias[st]:s.qb[st-2]),SB{}')
 s=s.replace('s.bf[seq&1].wait((seq/2)&1)','s.bf[seq&3].wait((seq/4)&1)')
 s=s.replace('s.bias[seq&1]+u*512+t*4','((seq&3)<2?s.bias[seq&3]:s.qb[(seq&3)-2])+u*512+t*4')
 s=s.replace('s.be[seq&1].arrive()','s.be[seq&3].arrive()')
 mark='  warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);\n }\n}'
 assert s.count(mark)==1
 s=s.replace(mark,'  __syncwarp();if(t%32==0)s.qd.arrive();\n'+mark)
 return s

def global_bias(s,copy_async=False):
 """Compare direct score loads vs warp cp.async while retaining Q/K/V TMA."""
 s=s.replace('int L;float scale; };','int L;float scale;float const* rawbias; };')
 s=s.replace('valid.data_ptr<int>(),L,float(scale)};', 'valid.data_ptr<int>(),L,float(scale),prepared.data_ptr<float>()};')
 a=s.index('  if(tid==Consumers){');b=s.index('  return;',a)
 if copy_async:
  producer='''  if(tid/32==Consumers/32){
   int lane=tid%32;
   #pragma unroll 1
   for(int g=0;g<6*48;++g){
    int st=g&1;
    if(g>=2){if(lane==0)s.be[st].wait(((g/2)-1)&1);__syncwarp();asm volatile("":::"memory");}
    float const* src=p.rawbias+(int64_t(h)*6*48+g)*2048;
    float* dst=s.bias[st];
    #pragma unroll
    for(int u=0;u<16;++u){
     uint32_t sd=cast_smem_ptr_to_uint(dst+(u*32+lane)*4);
     float const* gs=src+(u*32+lane)*4;
     asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(sd),"l"(gs):"memory");
    }
    asm volatile("cp.async.commit_group;":::"memory");
    asm volatile("cp.async.wait_group 0;":::"memory");
    __syncwarp();if(lane==0)s.bf[st].arrive();
   }
  }
'''
  s=s[:a]+producer+s[b:]
 else:
  s=s[:a]+s[b:]
  s=s.replace('    s.bf[seq&1].wait((seq/2)&1);asm volatile("":::"memory");\n','')
  s=s.replace('s.bias[seq&1]+u*512+t*4', 'p.rawbias+((int64_t(h)*6+qt)*48+seq)*2048+u*512+t*4')
  s=s.replace('   if(seq<48){__syncwarp();if(t%32==0)s.be[seq&1].arrive();}\n','')
 return s
