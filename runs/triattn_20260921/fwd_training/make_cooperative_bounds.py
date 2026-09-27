"""Range-certified softmax on the four-CTA single-score kernel. Keep fallback stable."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'cooperative_q1/fused.cu').read_text()
s=s.replace('cutlass::arch::ClusterTransactionBarrier qfull,', 'int fast[R];\n    cutlass::arch::ClusterTransactionBarrier qfull,')
s=s.replace('float* lse; int L;', 'float* lse; int L; float const* bounds;')
s=s.replace('    s.qfull.wait(0);', '    s.qfull.wait(0); asm volatile(""::: "memory");')
s=s.replace('    s.qfull.init(1);', '    s.qfull.init(1);\n    for(int r=0;r<Config::R;++r) s.fast[r]=1;')
prepass=r'''
// One K norm maximum per outer row; one live bias maximum per query.
__global__ void training_fwd_prepare_bounds(Element const* k,Element const* bias,float* bounds,int L) {
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,h=blockIdx.y,row=blockIdx.x;
  __shared__ float temp[8];
  if(blockIdx.z==0) {
    float km=0.f;
    for(int key=warp;key<L;key+=8) {
      float x=float(k[(int64_t(row)*L+key)*128+h*32+lane]);
      float n=x*x;
      for(int d=16;d;d/=2)n+=__shfl_xor_sync(0xffffffffu,n,d);
      km=fmaxf(km,n);
    }
    if(lane==0)temp[warp]=km;
    __syncthreads();
    if(tid==0) {
      float mx=0.f;for(int w=0;w<8;++w)mx=fmaxf(mx,temp[w]);
      bounds[h*L+row]=sqrtf(mx)*(1.f+1e-5f)+1e-6f;
    }
  } else {
    float bm=-INFINITY;int count=0;
    for(int key=tid;key<L;key+=256) {
      float b=float(bias[(int64_t(h)*L+row)*L+key]);
      if(b>-1e30f){bm=fmaxf(bm,b);++count;}
    }
    for(int d=16;d;d/=2) {
      bm=fmaxf(bm,__shfl_xor_sync(0xffffffffu,bm,d));
      count+=__shfl_xor_sync(0xffffffffu,count,d);
    }
    __shared__ int counts[8];
    if(lane==0){temp[warp]=bm;counts[warp]=count;}
    __syncthreads();
    if(tid==0) {
      float mx=-INFINITY;int n=0;
      for(int w=0;w<8;++w){mx=fmaxf(mx,temp[w]);n+=counts[w];}
      // Single-key rows retain the original LSE arithmetic and zero adjoints.
      bounds[4*L+h*L+row]=n>1?mx:-INFINITY;
    }
  }
}
'''
s=s.replace('template<bool Single> __global__',prepass+'\ntemplate<bool Single> __global__',1)
anchor='    auto qa=st.partition_fragment_A(sq);'
norm=r'''
    float shifts[2];bool eligible=true;
    float kn=p.bounds[h*L+row];
    uint32_t qptr=cast_smem_ptr_to_uint(s.q[r].data());
    #pragma unroll
    for(int mi=0;mi<2;++mi) {
      int qr=get<0>(sc(2*mi));float qn=0.f;
      #pragma unroll
      for(int d=lane%4;d<32;d+=4) {
        uint32_t addr=qptr+as_position_independent_swizzle_layout(Config::QL{})(make_coord(qr,d))*2;
        uint16_t bits;asm volatile("ld.shared.b16 %0,[%1];":"=h"(bits):"r"(addr):"memory");
        float x=float(Element::bitcast(bits));qn+=x*x;
      }
      qn+=__shfl_xor_sync(0xffffffffu,qn,1);qn+=__shfl_xor_sync(0xffffffffu,qn,2);
      float A=SCALE*sqrtf(qn)*kn;
      float bm=p.bounds[4*L+h*L+qt*64+qr];
      float ub=(bm+A)*LOG2E;
      shifts[mi]=ceilf(ub)+2.f;
      eligible=eligible && isfinite(ub) && (2.f*A*LOG2E<64.f) && fabsf(bm)<1e4f;
    }
    if(!__all_sync(0xffffffffu,eligible) && lane%32==0)atomicExch(&s.fast[r],0);
    cutlass::arch::NamedBarrier::sync(128,r+1);
    bool fast=s.fast[r]!=0;
    if(fast){m[0]=shifts[0];m[1]=shifts[1];l[0]=l[1]=0.f;}
'''
assert s.count(anchor)==1
s=s.replace(anchor,anchor+norm)
s=s.replace('mx[mi]=fmaxf(mx[mi],score(x));','if(!fast) mx[mi]=fmaxf(mx[mi],score(x));')
s=s.replace('''          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync''','''          if(fast){alpha[mi]=1.f;continue;}
          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync''')
s=s.replace('l[mi]=l[mi]*alpha[mi]+ls[mi];','if(fast)l[mi]+=ls[mi];else l[mi]=l[mi]*alpha[mi]+ls[mi];')
s=s.replace('for(int x=0;x<size(out);++x) out(x)*=alpha[(x%4)/2];','for(int x=0;x<size(out);++x) if(!fast)out(x)*=alpha[(x%4)/2];')
s=s.replace('  Config::Params p{make_q(q),make_q(k),make_q(v),tb,to,lse.data_ptr<float>(),L};', '''  auto bounds=torch::empty({2,4,L},q.options().dtype(torch::kFloat32));
  training_fwd_prepare_bounds<<<dim3(L,4,2),256,0,at::cuda::getCurrentCUDAStream()>>>((Element const*)k.data_ptr(),(Element const*)b.data_ptr(),bounds.data_ptr<float>(),L);
  Config::Params p{make_q(q),make_q(k),make_q(v),tb,to,lse.data_ptr<float>(),L,bounds.data_ptr<float>()};''')
out=root/'cooperative_bounds';out.mkdir(exist_ok=True)
(out/'fused.cu').write_text(s)
