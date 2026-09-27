// Inference front/back envelopes. No mean/rstd or backward activation buffers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
using B = __nv_bfloat16;
constexpr int VL=16;
__device__ __forceinline__ float as_float(float v){return v;}
__device__ __forceinline__ float as_float(B v){return __bfloat162float(v);}
__device__ __forceinline__ float sum(float v) {
  #pragma unroll
  for(int delta=4;delta;delta/=2)v+=__shfl_xor_sync(0xffffffff,v,delta);
  return v;
}
template<bool Ending,class G>
__global__ void inference_ln_bias(const B* x,const G* gamma,const G* beta,const B* wb,
                                  const bool* mask,B* z,B* bias,int L,float eps) {
  int lane=threadIdx.x%8,r=blockIdx.x*32+threadIdx.x/8;
  if(r>=L*L)return;
  int i=r/L,j=r%L,src=Ending?j*L+i:r;
  float v[VL],mu=0.f;
  uint4 input[VL/8];
  #pragma unroll
  for(int j=0;j<VL/8;++j)input[j]=*reinterpret_cast<const uint4*>(x+src*128+lane*VL+j*8);
  B* xp=reinterpret_cast<B*>(input);
  #pragma unroll
  for(int t=0;t<VL;++t){v[t]=__bfloat162float(xp[t]);mu+=v[t];}
  mu=sum(mu)*(1.f/128);float var=0.f;
  #pragma unroll
  for(int t=0;t<VL;++t){v[t]-=mu;var+=v[t]*v[t];}
  float inv=rsqrtf(sum(var)*(1.f/128)+eps);
  uint4 packed[VL/8];B* zp=reinterpret_cast<B*>(packed);float rounded[VL];
  #pragma unroll
  for(int t=0;t<VL;++t){
    float norm=v[t]*inv;
    float yy=fmaf(norm,as_float(gamma[lane*VL+t]),as_float(beta[lane*VL+t]));
    zp[t]=__float2bfloat16_rn(yy);rounded[t]=__bfloat162float(zp[t]);
  }
  #pragma unroll
  for(int j=0;j<VL/8;++j)*reinterpret_cast<uint4*>(z+r*128+lane*VL+j*8)=packed[j];
  uint2 packed_bias;auto heads=reinterpret_cast<B*>(&packed_bias);
  #pragma unroll
  for(int h=0;h<4;++h){
    float b=0.f;
    #pragma unroll
    for(int t=0;t<VL;++t)b=fmaf(rounded[t],__bfloat162float(wb[h*128+lane*VL+t]),b);
    b=sum(b);
    heads[h]=__float2bfloat16_rn(mask && !mask[j]?-3.3895313892515355e38f:b);
  }
  if(lane==0)*reinterpret_cast<uint2*>(bias+r*4)=packed_bias;
}
template<bool Ending>
__global__ void inference_residual(const B* x,const B* out,B* y,int L){
  int p=(blockIdx.x*blockDim.x+threadIdx.x)*8;
  if(p>=L*L*128)return;
  int r=p/128,ch=p%128,src=Ending?(r%L)*L+r/L:r;
  uint4 av=*reinterpret_cast<const uint4*>(x+p);
  uint4 bv=*reinterpret_cast<const uint4*>(out+src*128+ch),yv;
  auto aa=reinterpret_cast<__nv_bfloat162*>(&av);
  auto bb=reinterpret_cast<__nv_bfloat162*>(&bv);
  auto yy=reinterpret_cast<__nv_bfloat162*>(&yv);
  #pragma unroll
  for(int k=0;k<4;++k){
    float2 af=__bfloat1622float2(aa[k]),bf=__bfloat1622float2(bb[k]);
    yy[k]=__floats2bfloat162_rn(af.x+bf.x,af.y+bf.y);
  }
  *reinterpret_cast<uint4*>(y+p)=yv;
}
void check(torch::Tensor x){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() &&
              x.dim()==4 && x.size(0)==1 && x.size(1)==x.size(2) && x.size(3)==128,"B1 L L C128 BF16 required");
}
std::vector<torch::Tensor> front(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,
                                torch::Tensor wb,torch::Tensor mask,double eps,bool ending){
  check(x);c10::cuda::CUDAGuard guard(x.device());int L=x.size(1);
  for(auto w:{gamma,beta,wb})TORCH_CHECK(w.device()==x.device() && w.is_contiguous(),"invalid weight");
  TORCH_CHECK(wb.scalar_type()==x.scalar_type() && gamma.scalar_type()==beta.scalar_type() &&
              (gamma.scalar_type()==torch::kFloat32 || gamma.scalar_type()==torch::kBFloat16),"invalid weight dtype");
  TORCH_CHECK(gamma.numel()==128 && beta.numel()==128 && wb.sizes()==torch::IntArrayRef({4,128}),"invalid weight shape");
  TORCH_CHECK(mask.device()==x.device() && mask.scalar_type()==torch::kBool && mask.is_contiguous() &&
              (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({1,L})),"invalid mask");
  auto z=torch::empty_like(x),bias=torch::empty({1,L,L,4},x.options());
  auto run=[&](auto tag){
    using G=decltype(tag);auto fn=ending?inference_ln_bias<true,G>:inference_ln_bias<false,G>;
    fn<<<(L*L+31)/32,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (B*)x.data_ptr(),(G*)gamma.data_ptr(),(G*)beta.data_ptr(),(B*)wb.data_ptr(),
      mask.numel()?mask.data_ptr<bool>():nullptr,(B*)z.data_ptr(),(B*)bias.data_ptr(),L,float(eps));
  };
  if(gamma.scalar_type()==torch::kFloat32)run(float{});else run(B{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {z,bias};
}
torch::Tensor post(torch::Tensor x,torch::Tensor out,bool ending){
  check(x);check(out);TORCH_CHECK(x.device()==out.device() && x.sizes()==out.sizes(),"invalid output");
  c10::cuda::CUDAGuard guard(x.device());int L=x.size(1);auto y=torch::empty_like(x);
  auto fn=ending?inference_residual<true>:inference_residual<false>;
  fn<<<(L*L*16+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>((B*)x.data_ptr(),(B*)out.data_ptr(),(B*)y.data_ptr(),L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("bias_head_last",[](){return true;});m.def("front",&front);m.def("post",&post);m.def("smem",[](){return 0;});}
