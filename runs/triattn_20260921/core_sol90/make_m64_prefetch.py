"""Two score buffers, bias prefetch, fixed-length unrolled consumer stream."""
def transform(s, scaled=False):
 s=s.replace('p.L','768').replace('p.scale','0x1.6a09e6p-3f')
 s=s.replace('using namespace SOL_NAMESPACE;int L=q.size(-2);','using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 default scale required");')
 # Move shared bias reads a full body ahead of their consuming QK.
 a=s.index('  if(seq<nk){s.bf[seq%4].wait')
 b=s.index('  auto sk_all=',a)
 reads=s[a:b]
 s=s[:a]+s[b:]
 pos=s.index(' auto issue_qk=')
 s=s[:pos]+' auto init_score=[&](auto& sc,int seq) __attribute__((always_inline)) {\n'+reads+' };\n'+s[pos:]
 s=s.replace('   issue_qk(score,seq);','   init_score(score,seq);issue_qk(score,seq);')
 s=s.replace('  issue_qk(score,0);','  init_score(score,0);issue_qk(score,0);')
 s=s.replace('  issue_qk(next_score,1);','  init_score(next_score,1);issue_qk(next_score,1);')
 s=s.replace('issue_pv(score,0);\n  auto step=', 'issue_pv(score,0);init_score(score,2);\n  auto step=')
 s=s.replace('   issue_pv(sc,seq);','   issue_pv(sc,seq);init_score(sc,seq+2);')
 # Full unroll keeps asynchronous groups out of a loop backedge.
 s=s.replace('#pragma unroll 1\n  for(;seq+5<nk;seq+=6)', '#pragma unroll\n  for(;seq+5<nk;seq+=6)')
 s=s.replace('step(_0{},seq+5);drain();','step(_0{},seq+5);')
 s=s.replace('#pragma unroll 1\n  for(;seq<nk-1;seq+=2)', '#pragma unroll\n  for(;seq<nk-1;seq+=2)')
 s=s.replace('step(_0{},seq+1);drain();','step(_0{},seq+1);')
 if not scaled:return s
 s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
 marker=' QK qk;PV pv;LS ls;'
 s=s.replace(marker,''' using QElement=std::conditional_t<Safe,Element,cutlass::half_t>;
 using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<QElement,QElement,float,Shape<_64,Int<N>,_32>>()));
 using QR=decltype(make_tiled_mma(GMMA::rs_op_selector<QElement,QElement,float,Shape<_64,Int<N>,_32>>()));
'''+marker)
 s=s.replace(' auto sq=make_tensor(make_smem_ptr(s.q[wg]),SQ{});auto qa=tq.partition_fragment_A(sq);', ''' auto sq=make_tensor(make_smem_ptr(reinterpret_cast<QElement*>(s.q[wg])),SQ{});
 QR qr;auto tqreg=qr.get_thread_slice(t);auto qa=tqreg.partition_fragment_A(sq);''')
 pos=s.index(' auto init_score=')
 s=s[:pos]+''' auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,QElement>{},qr);
 auto qtcopy=qcopy.get_thread_slice(t);auto qs=qtcopy.partition_S(sq);auto qd=qtcopy.retile_D(qa);
 copy(qcopy,qs,qd);
 if constexpr(!Safe){
  auto words=recast<uint32_t>(qd);
  #pragma unroll
  for(int j=0;j<size(words);j++){
   unsigned u=words(j);
   auto pair=__floats2half2_rn(__uint_as_float(u<<16)*c,__uint_as_float(u&0xffff0000u)*c);
   words(j)=reinterpret_cast<unsigned const&>(pair);
  }
 }
 warpgroup_fence_operand(qa);
'''+s[pos:]
 s=s.replace('make_smem_ptr(s.k[st][wg])','make_smem_ptr(reinterpret_cast<QElement*>(s.k[st][wg]))')
 s=s.replace('gemm(qk,qa(', 'gemm(qr,qa(')
 s=s.replace('if constexpr(Safe || decltype(seed)::value)', 'if constexpr(Safe)')
 s=s.replace('else sr(row,col)=ex2(fmaf(x,c,-mx[row]));','else sr(row,col)=ex2(x);')
 pos=s.index('__global__ void prepare(')
 s=s[:pos]+'''__global__ void scale_hot_bias(float const* src,float* dst,int n,float c){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)dst[i]=fmaf(src[i],c,-64.f);
}
'''+s[pos:]
 tk=' auto tk=make_tma_copy(SM90_TMA_LOAD{},g(k),SK{},Shape<Int<LN>,_32>{},_1{});'
 s=s.replace(tk,tk+'''
 auto kh=k.to(at::kHalf);
 auto tkh=make_tma_copy(SM90_TMA_LOAD{},g(kh),SK{},Shape<Int<LN>,_32>{},_1{});
 auto bh=at::empty_like(prepared);
 scale_hot_bias<<<4096,256,0,stream>>>(prepared.data_ptr<float>(),bh.data_ptr<float>(),prepared.numel(),float(scale)*1.4426950408889634f);
''')
 tb=next(line for line in s.splitlines(True) if line.startswith(' auto tb='))
 s=s.replace(tb,tb+tb.replace('auto tb=','auto tbh=').replace('prepared.data_ptr<float>()','bh.data_ptr<float>()'))
 s=s.replace(' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);',' auto ph=p;ph.k=tkh;ph.bias=tbh;\n attention<false><<<ct,Threads,sizeof(Shared),stream>>>(ph);')
 return s
