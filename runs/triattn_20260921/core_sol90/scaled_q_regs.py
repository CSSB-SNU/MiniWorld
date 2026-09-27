"""Unqualified experiment: RS QK with scaled FP16 Q and converted shared K.

Hot QK is in log2 units. SAFE reloads the original BF16 inputs and original
bias. V, probability packing, and output remain BF16. No serving import.
"""
def transform(name,s):
 if name=='triattn_m1.py':
  s=s.replace('    qkv = []','    assert scale > 0 and math.isfinite(scale), "positive finite scale required"\n    qkv = []')
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
  s=s.replace('using Element = cutlass::bfloat16_t;','using Element = cutlass::bfloat16_t;\n    using QElement = std::conditional_t<kSafe, Element, cutlass::half_t>;')
  s=s.replace('GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>','GMMA::ss_op_selector<QElement, QElement, float, Shape<_64, Int<CW>, Int<kHeadDim>>>')
  s=s.replace('GMMA::rs_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>','GMMA::rs_op_selector<QElement, QElement, float, Shape<_64, Int<CW>, Int<kHeadDim>>>')
  s=s.replace('kQinRegs = (kFlags_ & 1048576) != 0','kQinRegs = !kSafe')
  a=s.index('    // ================================================= CONSUMERS')
  front,body=s[:a],s[a:]
  body='    using QElement = typename T::QElement;\n'+body
  body=body.replace('make_smem_ptr(shared.smem_q.data())','make_smem_ptr(reinterpret_cast<QElement*>(shared.smem_q.data()))')
  body=body.replace('make_smem_ptr(shared.smem_k.data() + cwg * T::kStageElemsK + zoff)','make_smem_ptr(reinterpret_cast<QElement*>(shared.smem_k.data()) + cwg * T::kStageElemsK + zoff)')
  body=body.replace('Copy_Atom<SM75_U32x4_LDSM_N, Element>', 'Copy_Atom<SM75_U32x4_LDSM_N, QElement>')
  line='                cute::copy(smem_tiled_copy_q, tQsQ, tQrQ);'
  assert body.count(line)==1
  body=body.replace(line,line+'''
                auto words=recast<uint32_t>(tQrQ);
                #pragma unroll
                for(int n=0;n<size(words);++n){
                    uint32_t u=words(n);
                    float a=__uint_as_float(u<<16), b=__uint_as_float(u&0xffff0000u);
                    __half2 packed=__floats2half2_rn(a*(params.scale*kLog2e),b*(params.scale*kLog2e));
                    words(n)=reinterpret_cast<uint32_t const&>(packed);
                }
''')
  # Each consumer owns its K row. Convert only after the producer's complete
  # transaction, then make generic writes visible to asynchronous MMA reads.
  body=body.replace('pipe_kv.wait_full(', 'wait_kv_ready(')
  pos=body.index('    // register copies of this warpgroup')
  helper='''    auto wait_kv_ready = [&](int st,uint32_t phase) __attribute__((always_inline)) {
        pipe_kv.wait_full(st,phase);
        if constexpr(!kSafe){
            uint32_t* words=reinterpret_cast<uint32_t*>(shared.smem_k.data()+st*T::kStageElemsK);
            #pragma unroll
            for(int j=0;j<T::kStageElemsK/2/128;++j){
                int index=t128+j*128; uint32_t u=words[index];
                __half2 packed=__floats2half2_rn(__uint_as_float(u<<16),__uint_as_float(u&0xffff0000u));
                words[index]=reinterpret_cast<uint32_t const&>(packed);
            }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(128,uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier)+cwg);
        }
    };
'''
  body=body[:pos]+helper+body[pos:]
  body=body.replace('bool need_seed = true;', 'bool need_seed = kSafe;')
  body=body.replace('if constexpr (kSeedHere) {','if constexpr (kSeedHere && kSafe) {')
  exp='s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]));'
  assert body.count(exp)==1
  body=body.replace(exp,'if constexpr(kSafe) { '+exp+' } else { s_rc(mi,ni)=ex2_approx(s_rc(mi,ni)); }')
  la=body.index('    auto l_check =');lb=body.index('    // dep:',la)
  section=body[la:lb].replace('if constexpr (!kSafe)', 'if constexpr (false)')
  body=body[:la]+section+body[lb:]
  body=body.replace('!(chk > 0.f && chk < INFINITY)', '!(chk > 0x1p-84f && chk < 0x1p-44f)')
  s=front+body
 if name=='m1_binding.cu':
  pos=s.index('// hot pass (flags)')
  s=s[:pos]+'''// One small staged-bias transform per call; included in timing.
__global__ void scale_hot_bias(float const* src,float* dst,int64_t n,float c){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
        dst[i]=fmaf(src[i],c,-64.f);
}
'''+s[pos:]
  a=s.index('    Args a{q, k, v, bias_staged');b=s.index(', tr};',a)+len(', tr};')
  decl=s[a:b]
  prefix='''    auto bias_hot=torch::empty_like(bias_staged);
    scale_hot_bias<<<std::min<int64_t>(4096,(bias_staged.numel()+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(bias_staged.data_ptr<float>(),bias_hot.data_ptr<float>(),bias_staged.numel(),float(scale)*1.4426950408889634f);
'''
  s=s[:a]+prefix+decl+'\n'+decl.replace('Args a{','Args hot_a{').replace('bias_staged','bias_hot')+s[b:]
  s=s.replace('    hot->second.run(a);','    hot_a.force_fix=a.force_fix;\n    hot->second.run(hot_a);')
 return s
