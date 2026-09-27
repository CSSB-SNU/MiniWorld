"""Numerical experiment on installed9365: externally scaled FP16 Q/K.

Keep original BF16 operands for SAFE. Global conversion is timed. The fast
path uses log2-scaled Q, staged log2 bias minus64, and EX2 without per-score
FMA, seeding or rescaling. No serving import or precision claim.
"""

def transform(name,s,full_q):
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
  s=s.replace('    using Element = cutlass::bfloat16_t;', '    using Element = cutlass::bfloat16_t;\n    using QElement = std::conditional_t<!kSafe && (kFlags_ & (1073741824|536870912)),cutlass::half_t,Element>;')
  s=s.replace('GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>', 'GMMA::ss_op_selector<QElement, QElement, float, Shape<_64, Int<CW>, Int<kHeadDim>>>')
  s=s.replace('GMMA::rs_op_selector<Element, Element, float, Shape<_64, Int<CW>, Int<kHeadDim>>>', 'GMMA::rs_op_selector<QElement, QElement, float, Shape<_64, Int<CW>, Int<kHeadDim>>>')
  pos=s.index('    typename T::TiledMmaQK tiled_mma_qk;')
  s=s[:pos]+'    using QElement=typename T::QElement;\n'+s[pos:]
  s=s.replace('make_smem_ptr(shared.smem_q.data()), typename T::SmemLayoutQ{}', 'make_smem_ptr(reinterpret_cast<QElement*>(shared.smem_q.data())), typename T::SmemLayoutQ{}')
  s=s.replace('make_smem_ptr(shared.smem_k.data() + cwg * T::kStageElemsK + zoff)', 'make_smem_ptr(reinterpret_cast<QElement*>(shared.smem_k.data()) + cwg * T::kStageElemsK + zoff)')
  s=s.replace('Copy_Atom<SM75_U32x4_LDSM_N, Element>', 'Copy_Atom<SM75_U32x4_LDSM_N, QElement>')
  s=s.replace('bool need_seed = true;', 'bool need_seed = !kFast;')
  s=s.replace('if constexpr (kSeedHere) {', 'if constexpr (kSeedHere && !kFast) {')
  a=s.index('        if constexpr(kFast){\n            #pragma unroll\n            for(int mi=0;mi<kNRows;')
  b=s.index('        }else{',a)
  s=s[:a]+'''        if constexpr(kFast){
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=0;ni<kNC;ni+=4){
                    asm volatile(
                    "ex2.approx.ftz.f32 %0,%0;\\n"
                    "ex2.approx.ftz.f32 %1,%1;\\n"
                    "ex2.approx.ftz.f32 %2,%2;\\n"
                    "ex2.approx.ftz.f32 %3,%3;\\n"
                    : "+f"(s_rc(mi,ni)), "+f"(s_rc(mi,ni+1)), "+f"(s_rc(mi,ni+2)), "+f"(s_rc(mi,ni+3)));
                }
            }
'''+s[b:]
  a=s.index('    auto l_check =');b=s.index('    // dep:',a)
  s=s[:a]+s[a:b].replace('if constexpr (!kSafe)', 'if constexpr (!kSafe && !kFast)')+s[b:]
  s=s.replace('!(chk > 0.f && chk < INFINITY)', '!(kFast ? (chk > 0x1p-84f && chk < 0x1p-44f) : (chk > 0.f && chk < INFINITY))')
  # Producer TMA descriptors/storage remain BF16-typed raw16-bit copies;
  # only consumer interpretation changes to half for the fast instantiation.
  pos=s.index('    using QElement=typename T::QElement;')
  s=s[:pos].replace('make_smem_ptr(reinterpret_cast<QElement*>(shared.smem_q.data()))','make_smem_ptr(shared.smem_q.data())')+s[pos:]
  if full_q:
   assert s.count('if constexpr(hh==0){')==1
   s=s.replace('if constexpr(hh==0){', 'if constexpr(true){')
   old='''            auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(a1);'''
   assert s.count(old)==2,s.count(old)
   s=s.replace(old,'            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
 if name=='m1_binding.cu':
  s=s.replace('#include <cuda_bf16.h>', '#include <cuda_bf16.h>\n#include <cuda_fp16.h>')
  s=s.replace('#include <cuda_fp16.h>\n#include <cuda_fp16.h>','#include <cuda_fp16.h>')
  pos=s.index('// hot pass (flags)')
  s=s[:pos]+r'''
// Preparation is included in core timing. Non-roundtrippable conversions
// become NaNs so the existing fix-list path uses original BF16 operands.
__device__ __forceinline__ uint32_t scale_pair_checked(uint32_t bits,float scale,float inverse){
 float a=__uint_as_float(bits<<16),b=__uint_as_float(bits&0xffff0000u);
 __half2 h=__floats2half2_rn(a*scale,b*scale);
 uint32_t result=reinterpret_cast<uint32_t const&>(h);
 __nv_bfloat16 ba=__float2bfloat16_rn(__low2float(h)*inverse);
 __nv_bfloat16 bb=__float2bfloat16_rn(__high2float(h)*inverse);
 if(reinterpret_cast<uint16_t const&>(ba)!=(bits&0xffffu))result=(result&0xffff0000u)|0x7e00u;
 if(reinterpret_cast<uint16_t const&>(bb)!=(bits>>16))result=(result&0xffffu)|0x7e000000u;
 return result;
}
__global__ void prepare_scaled_qk(uint32_t const* q,uint32_t const* k,uint32_t* qo,uint32_t* ko,int64_t pairs,float c,float inv){
 for(int64_t x=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;x<pairs;x+=int64_t(gridDim.x)*blockDim.x){
  qo[x]=scale_pair_checked(q[x],c,inv);ko[x]=scale_pair_checked(k[x],1.f,1.f);
 }
}
__global__ void prepare_scaled_bias(float const* src,float* dst,int64_t count,float c){
 for(int64_t x=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;x<count;x+=int64_t(gridDim.x)*blockDim.x)dst[x]=fmaf(src[x],c,-64.f);
}
'''+s[pos:]
  marker='    TORCH_CHECK(hot != table().end() && safe != table().end(),'
  pos=s.index(marker)
  s=s[:pos]+'''    // The experimental converter uses flat contiguous storage.
    if(hot!=table().end() && (hot->first==1073741824 || hot->first==536870912) && (!q.is_contiguous() || !k.is_contiguous()))hot=table().find(0);
'''+s[pos:]
  a=s.index('    Args a{q, k, v, bias_staged');b=s.index(', tr};',a)+len(', tr};')
  args=s[a:b]
  old='    hot->second.run(a);'
  assert s.count(old)==1
  new='''    if(hot->first==1073741824 || hot->first==536870912){
        auto q_hot=torch::empty_like(q),k_hot=torch::empty_like(k),bias_hot=torch::empty_like(bias_staged);
        float c=float(scale)*1.4426950408889634f;
        prepare_scaled_qk<<<4096,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<uint32_t const*>(q.data_ptr()),reinterpret_cast<uint32_t const*>(k.data_ptr()),reinterpret_cast<uint32_t*>(q_hot.data_ptr()),reinterpret_cast<uint32_t*>(k_hot.data_ptr()),q.numel()/2,c,1.f/c);
        prepare_scaled_bias<<<std::min<int64_t>(4096,(bias_staged.numel()+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(bias_staged.data_ptr<float>(),bias_hot.data_ptr<float>(),bias_staged.numel(),c);
'''+args.replace('Args a{q, k, v, bias_staged','Args hot_a{q_hot, k_hot, v, bias_hot')+'''
        hot_a.force_fix=a.force_fix;hot->second.run(hot_a);
    }else hot->second.run(a);'''
  s=s.replace(old,new)
 return s
