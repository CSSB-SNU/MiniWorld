"""Scaled FP16 RS QK with three scores, fully unrolled L768 hot stream."""
def transform(s):
 from make_wide import transform as add_denominator
 s=add_denominator(s)
 s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
 marker=' QK qk;PV pv;LS ls;'
 local=''' using QElement=std::conditional_t<Safe,Element,cutlass::half_t>;
 using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<QElement,QElement,float,Shape<_64,Int<N>,_32>>()));
 using QR=decltype(make_tiled_mma(GMMA::rs_op_selector<QElement,QElement,float,Shape<_64,Int<N>,_32>>()));
'''
 assert s.count(marker)==1
 s=s.replace(marker,local+marker)
 s=s.replace('make_smem_ptr(s.q[wg])','make_smem_ptr(reinterpret_cast<QElement*>(s.q[wg]))')
 s=s.replace('make_smem_ptr(s.k[st][wg])','make_smem_ptr(reinterpret_cast<QElement*>(s.k[st][wg]))')
 s=s.replace('Copy_Atom<SM75_U32x4_LDSM_N,Element>', 'Copy_Atom<SM75_U32x4_LDSM_N,QElement>')
 s=s.replace('  copy(qcopy,qs,qd);','''  copy(qcopy,qs,qd);
  if constexpr(!Safe){
   auto words=recast<uint32_t>(qd);
   #pragma unroll
   for(int j=0;j<size(words);j++){
    unsigned u=words(j);
    __half2 pair=__floats2half2_rn(__uint_as_float(u<<16)*c,__uint_as_float(u&0xffff0000u)*c);
    words(j)=reinterpret_cast<unsigned const&>(pair);
   }
  }''')
 s=s.replace('if constexpr(Safe || decltype(seed)::value)', 'if constexpr(Safe)')
 s=s.replace('else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));','else sr(row,col)=ex2(x);')
 s=s.replace('#pragma unroll 1\n  for(;seq+5<2*nk;seq+=6)', '#pragma unroll\n  for(;seq+5<2*nk;seq+=6)')
 s=s.replace('step(seq+5,_1{},_1{});drain();','step(seq+5,_1{},_1{});')
 s=s.replace('if(seq<2*nk){step(seq,_0{},_2{});step(seq+1,_1{},_0{});drain();}', 'if(seq<2*nk){step(seq,_0{},_2{});step(seq+1,_1{},_0{});}')
 s=s.replace('if(seq+2<2*nk){step(seq+2,_0{},_1{});step(seq+3,_1{},_2{});drain();}', 'if(seq+2<2*nk){step(seq+2,_0{},_1{});step(seq+3,_1{},_2{});}')
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
