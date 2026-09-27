"""Lossless BF16-code bias storage, checked against the original staged FP32."""
def transform(name,s):
 def rep(a,b,n=1):
  nonlocal s
  assert s.count(a)==n,(name,a,s.count(a),n)
  s=s.replace(a,b)
 if name=='triattn_m1_sm90.cuh':
  rep('using Element = cutlass::bfloat16_t;', '''using Element = cutlass::bfloat16_t;
    static constexpr bool kCompressed = !kSafe && (kFlags_ & 1073741824);
    using BiasElement = std::conditional_t<kCompressed,Element,float>;''')
  a=s.index('    using TMA_B =');b=s.index('// ABLATION',a)
  s=s[:a]+s[a:b].replace('static_cast<float const*>','static_cast<BiasElement const*>')+s[b:]
  rep('static constexpr uint32_t kBytesHalf = kHalfElems * sizeof(float);','static constexpr uint32_t kBytesHalf = kHalfElems * sizeof(BiasElement);')
  rep('cute::array_aligned<float, kSlotElems * kSlotsB, 1024> smem_bias;', 'cute::array_aligned<BiasElement, kSlotElems * kSlotsB, 1024> smem_bias;')
  rep('int const* rowkc1;', 'int const* rowkc1;\n        int const* biasbad; float bias_inv_scale;')
  rep('bool bad = (((kind & 1) != 0) && !kMaskInKernel) || (!kSafe && params.force_fix != 0);', '''bool bad = (((kind & 1) != 0) && !kMaskInKernel) || (!kSafe && params.force_fix != 0);
    if constexpr(T::kCompressed)bad |= params.biasbad[bh*6+qtile]!=0;''')
  rep('float const* bias_thread = shared.smem_bias.data()', 'typename T::BiasElement const* bias_thread = shared.smem_bias.data()')
  old='''                float4 const f4 = *reinterpret_cast<float4 const*>(bias_thread + c * T::kSlotElems + (hh * 4 + u) * 512);
                acc(4 * u + 0) = f4.x; acc(4 * u + 1) = f4.y; acc(4 * u + 2) = f4.z; acc(4 * u + 3) = f4.w;'''
  rep(old,'''                if constexpr(T::kCompressed){
                    uint2 const packed=*reinterpret_cast<uint2 const*>(bias_thread+c*T::kSlotElems+(hh*4+u)*512);
                    acc(4*u+0)=__uint_as_float(packed.x<<16)*params.bias_inv_scale;
                    acc(4*u+1)=__uint_as_float(packed.x&0xffff0000u)*params.bias_inv_scale;
                    acc(4*u+2)=__uint_as_float(packed.y<<16)*params.bias_inv_scale;
                    acc(4*u+3)=__uint_as_float(packed.y&0xffff0000u)*params.bias_inv_scale;
                }else{
'''+old+'''
                }''')
 elif name=='launch_m1.cuh':
  rep('int force_fix = 0;', 'torch::Tensor const* bias_codes = nullptr;\n    int const* biasbad = nullptr;\n    int force_fix = 0;')
  rep('Tensor mB = make_tensor(make_gmem_ptr(a.bias.data_ptr<float>()), shape_b, stride_b);', '''auto const& biasbuf=T::kCompressed ? *a.bias_codes : a.bias;
    Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<typename T::BiasElement const*>(biasbuf.data_ptr())), shape_b, stride_b);
    Tensor mBfloat = make_tensor(make_gmem_ptr(a.bias.data_ptr<float>()), shape_b, stride_b);''')
  rep('TMA_BT tma_bt = make_tma_copy(SM90_TMA_LOAD{}, mB,', 'TMA_BT tma_bt = make_tma_copy(SM90_TMA_LOAD{}, mBfloat,')
  rep('n_kcol, a.rowkc0, a.rowkc1};','n_kcol, a.rowkc0, a.rowkc1, a.biasbad, float(1.0/a.scale)};')
 elif name=='m1_binding.cu':
  rep('struct Entry {', '''// A single CTA writes every code and the validation bit for one bias q-tile.
__global__ void encode_bias(float const* src,__nv_bfloat16* dst,int* bad,int count,float scale,float inv_scale){
    int tile=blockIdx.x; bool failed=false;
    for(int x=threadIdx.x;x<count;x+=blockDim.x){
        int64_t ix=int64_t(tile)*count+x;float value=src[ix];
        __nv_bfloat16 code=__float2bfloat16_rn(value*scale);
        float restored=__bfloat162float(code)*inv_scale;
        failed |= restored!=value;dst[ix]=code;
    }
    int any=__syncthreads_or(failed);if(threadIdx.x==0)bad[tile]=any;
}

struct Entry {''')
  rep('    hot->second.run(a);', '''    torch::Tensor codes,biasbad;
    if(hot->first==1073741824){
        codes=torch::empty(bias_staged.sizes(),bias_staged.options().dtype(torch::kBFloat16));
        biasbad=torch::empty({q.size(0)*q.size(2)*6},q.options().dtype(torch::kInt32));
        encode_bias<<<biasbad.numel(),256,0,at::cuda::getCurrentCUDAStream()>>>(
            bias_staged.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(codes.data_ptr()),biasbad.data_ptr<int>(),
            bias_staged.size(2)*4096,float(scale),float(1.0/scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();a.bias_codes=&codes;a.biasbad=biasbad.data_ptr<int>();
    }
    hot->second.run(a);''')
 return s
