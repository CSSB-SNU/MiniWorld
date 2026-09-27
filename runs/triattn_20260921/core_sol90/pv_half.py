"""Isolated mixed precision PV experiment. SAFE and non-fast retain BF16/FP32."""
def transform(name,s,qregs=True):
 def rep(a,b,n=1):
  nonlocal s
  assert s.count(a)==n,(name,a,s.count(a),n)
  s=s.replace(a,b)
 if name=='triattn_m1_sm90.cuh':
  rep('using Element = cutlass::bfloat16_t;', '''using Element = cutlass::bfloat16_t;
    static constexpr bool kHalfPV = !kSafe && (kFlags_ & 1073741824);
    using PVElement = std::conditional_t<kHalfPV, cutlass::half_t, Element>;''')
  if qregs:rep('static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;', 'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || kHalfPV;')
  rep('using TiledMmaPV = decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));', '''using PVOp = std::conditional_t<kHalfPV,SM90_64x32x16_F16F16F16_RS<GMMA::Major::K,GMMA::Major::MN>,SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>>;
    using TiledMmaPV = decltype(make_tiled_mma(PVOp{}, AtomLayout{}));''')
  rep('using TiledMmaL  = decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, AtomLayout{}));', '''using LOp = std::conditional_t<kHalfPV,SM90_64x8x16_F32F16F16_RS<GMMA::Major::K,GMMA::Major::K>,SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>>;
    using TiledMmaL = decltype(make_tiled_mma(LOp{},AtomLayout{}));''')
  rep('using TMA_V = TMA_K;', '''using TMA_V = decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<PVElement const*>(nullptr)),ShapeQK{},StrideQK{}),
        take<0,2>(SmemLayoutV{}),make_shape(Int<kBlockN>{},Int<kHeadDim>{}),_1{}));''')
  rep('cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;', 'cute::array_aligned<PVElement, cute::cosize_v<SmemLayoutV>, 1024> smem_v;')
  rep('cute::array_aligned<Element, cute::cosize_v<SmemLayoutOnes>, 128> smem_ones;', 'cute::array_aligned<PVElement, cute::cosize_v<SmemLayoutOnes>, 128> smem_ones;')
  rep('int const* rowkc1;', 'int const* rowkc1;\n        int const* vbad;')
  rep('shared.smem_ones[idx] = Element(1.f);', 'shared.smem_ones[idx] = typename T::PVElement(1.f);')
  rep('make_tensor_like<Element>(make_tensor(accC[0].data()', 'make_tensor_like<typename T::PVElement>(make_tensor(accC[0].data()')
  rep('bool bad = (((kind & 1) != 0) && !kMaskInKernel) || (!kSafe && params.force_fix != 0);', '''bool bad = (((kind & 1) != 0) && !kMaskInKernel) || (!kSafe && params.force_fix != 0);
    if constexpr(T::kHalfPV)bad |= params.vbad[(b*768+min(i0+cwg,767))*4+h]!=0;''')
  rep('if constexpr (kPrmtPack) {', '''if constexpr(T::kHalfPV){
                __half2 h2=__floats2half2_rn(acc(2*pr),acc(2*pr+1));
                dst32(pr)=reinterpret_cast<uint32_t const&>(h2);
            } else if constexpr (kPrmtPack) {''')
  rep('constexpr float kShift = kSafe ? 0.f : 64.f;', 'constexpr float kShift = (kSafe || T::kHalfPV) ? 0.f : 64.f;')
  rep('constexpr uint32_t kLo = uint32_t(127 - 84) << 23, kWidth = (uint32_t(127 - 44) << 23) - kLo;', '''constexpr uint32_t kLo = uint32_t(127 + (T::kHalfPV ? -4 : -84)) << 23;
        constexpr uint32_t kWidth = (uint32_t(127 + (T::kHalfPV ? 12 : -44)) << 23) - kLo;''')
  rep('int k = int((lb >> 23) & 0xffu) - 127 + 64;', 'int k = int((lb >> 23) & 0xffu) - 127 + (T::kHalfPV ? -4 : 64);')
  rep('int const zo = params.zero * p;', 'int const zo = kFast ? 0 : params.zero * p;',2)
 elif name=='launch_m1.cuh':
  rep('int force_fix = 0;', 'torch::Tensor const* vhalf = nullptr;\n    int const* vbad = nullptr;\n    int force_fix = 0;')
  rep('Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), shape_qk, stride_of(a.v));', '''auto const& pv_v = T::kHalfPV ? *a.vhalf : a.v;
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<typename T::PVElement const*>(pv_v.data_ptr())), shape_qk, stride_of(pv_v));''')
  rep('n_kcol, a.rowkc0, a.rowkc1};', 'n_kcol, a.rowkc0, a.rowkc1, a.vbad};')
 elif name=='m1_binding.cu':
  rep('struct Entry {', '''// One CTA per pair-row/head. Any inexact V conversion requests the original SAFE path.
__global__ void convert_pv_half(__nv_bfloat16 const* v, int64_t vb,int64_t vn,int64_t vh,int64_t vs,
                               __half* out,int* bad,int N,int H,int S){
    int row=blockIdx.x, h=row%H,n=(row/H)%N,b=row/(N*H);
    bool failed=false;
    for(int x=threadIdx.x;x<S*32;x+=blockDim.x){
        float f=__bfloat162float(v[int64_t(b)*vb+int64_t(n)*vn+int64_t(h)*vh+int64_t(x/32)*vs+x%32]);
        __half y=__float2half_rn(f);out[int64_t(row)*S*32+x]=y;
        failed |= __half2float(y)!=f;
    }
    int any=__syncthreads_or(failed);
    if(threadIdx.x==0)bad[row]=any;
}

struct Entry {''')
  rep('    hot->second.run(a);', '''    torch::Tensor vhalf,vbad;
    if(hot->first==1073741824){
        vhalf=torch::empty(v.sizes(),v.options().dtype(torch::kFloat16));
        vbad=torch::empty({q.size(0)*q.size(1)*q.size(2)},q.options().dtype(torch::kInt32));
        convert_pv_half<<<vbad.numel(),256,0,at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<__nv_bfloat16 const*>(v.data_ptr()),v.stride(0),v.stride(1),v.stride(2),v.stride(3),
            reinterpret_cast<__half*>(vhalf.data_ptr()),vbad.data_ptr<int>(),q.size(1),q.size(2),q.size(3));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        a.vhalf=&vhalf;a.vbad=vbad.data_ptr<int>();
    }
    hot->second.run(a);''')
 return s
