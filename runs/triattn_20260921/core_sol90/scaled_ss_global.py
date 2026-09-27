"""Diagnostic for folding Q/K scaling into a future prologue; casts are timed."""
def transform(name,s):
 from scaled_producer import transform as producer
 s=producer(name,s)
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('        cutlass::arch::ClusterBarrier q_converted, k_converted[kStagesKV];','')
  a=s.index('        if constexpr(!kSafe){\n            shared.q_converted.init(2);')
  b=s.index('        T::PipeKV::init',a)
  s=s[:a]+s[b:]
  a=s.index('        if constexpr(!kSafe){\n            if(warp_idx_in_wg>=2){')
  b=s.index('        if constexpr (!kList) { break; }\n        }   // producer tile loop',a)
  s=s[:a]+s[b:]
  s=s.replace('if constexpr(kSafe)shared.barrier_q.wait(0);else shared.q_converted.wait(0);','shared.barrier_q.wait(0);')
  a=s.index('    auto wait_kv_ready =');b=s.index('    // register copies of this warpgroup',a)
  s=s[:a]+'''    auto wait_kv_ready = [&](int st,uint32_t phase) __attribute__((always_inline)) { pipe_kv.wait_full(st,phase); };
'''+s[b:]
 if name=='m1_binding.cu':
  s=s.replace('    auto bias_hot=torch::empty_like(bias_staged);','''    auto q_hot=(q.to(torch::kFloat32)*(float(scale)*1.4426950408889634f)).to(torch::kFloat16);
    auto k_hot=k.to(torch::kFloat16);
    auto bias_hot=torch::empty_like(bias_staged);''')
  s=s.replace('Args hot_a{q, k, v, bias_hot','Args hot_a{q_hot, k_hot, v, bias_hot')
 return s

def generic(s):
 s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
 s=s.replace(' QK qk;PV pv;LS ls;', ''' using QElement=std::conditional_t<Safe,Element,cutlass::half_t>;
 using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<QElement,QElement,float,Shape<_64,Int<N>,_32>>()));
 QK qk;PV pv;LS ls;''')
 s=s.replace('make_smem_ptr(s.q[wg])','make_smem_ptr(reinterpret_cast<QElement*>(s.q[wg]))')
 s=s.replace('make_smem_ptr(s.k[st][wg])','make_smem_ptr(reinterpret_cast<QElement*>(s.k[st][wg]))')
 s=s.replace('if constexpr(Safe || decltype(seed)::value)', 'if constexpr(Safe)')
 s=s.replace('else sr(row,col)=ex2(fmaf(x,c,-mx[row]));','else sr(row,col)=ex2(x);')
 pos=s.index('__global__ void prepare(')
 s=s[:pos]+'''__global__ void scale_hot_bias(float const* src,float* dst,int n,float c){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)dst[i]=fmaf(src[i],c,-64.f);
}
'''+s[pos:]
 tk=' auto tk=make_tma_copy(SM90_TMA_LOAD{},g(k),SK{},Shape<Int<LN>,_32>{},_1{});'
 s=s.replace(tk,tk+'''
 auto qh=(q.to(at::kFloat)*(float(scale)*1.4426950408889634f)).to(at::kHalf);
 auto kh=k.to(at::kHalf);
 auto tqh=make_tma_copy(SM90_TMA_LOAD{},g(qh),SQ{},Shape<_64,_32>{},_1{});
 auto tkh=make_tma_copy(SM90_TMA_LOAD{},g(kh),SK{},Shape<Int<LN>,_32>{},_1{});
 auto bh=at::empty_like(prepared);
 scale_hot_bias<<<4096,256,0,stream>>>(prepared.data_ptr<float>(),bh.data_ptr<float>(),prepared.numel(),float(scale)*1.4426950408889634f);
''')
 tb=next(line for line in s.splitlines(True) if line.startswith(' auto tb='))
 s=s.replace(tb,tb+tb.replace('auto tb=','auto tbh=').replace('prepared.data_ptr<float>()','bh.data_ptr<float>()'))
 s=s.replace(' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);',' auto ph=p;ph.q=tqh;ph.k=tkh;ph.bias=tbh;\n attention<false><<<ct,Threads,sizeof(Shared),stream>>>(ph);')
 return s
