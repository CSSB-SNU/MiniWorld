#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "half2_exp_probe.cuh"
template<int D> __global__ void map_exp(float const* x,float* y,int count){
 int i=2*(blockIdx.x*blockDim.x+threadIdx.x);if(i<count){
  float2 out=half2_exp_pair<D>(x[i],i+1<count?x[i+1]:0.f);y[i]=out.x;if(i+1<count)y[i+1]=out.y;
 }
}
template<int D> __global__ __launch_bounds__(128) void register_probe(float* out,int loops){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;float x[16];
 #pragma unroll
 for(int j=0;j<16;++j)x[j]=-64.f-float(tid%13)*.013f-float(j)*.03125f;
 #pragma unroll 1
 for(int it=0;it<loops;++it){
  #pragma unroll
  for(int j=0;j<16;j+=2){
   float2 y=half2_exp_pair<D>(x[j],x[j+1]);
   x[j]=__uint_as_float((__float_as_uint(y.x)&0x003fffffu)|0xc2800000u);
   x[j+1]=__uint_as_float((__float_as_uint(y.y)&0x003fffffu)|0xc2800000u);
  }
 }
 float total=0;for(int j=0;j<16;++j)total+=x[j];out[tid]=total;
}
at::Tensor probe_exp(at::Tensor x,int degree){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kFloat,"contiguous CUDA float required");
 c10::cuda::CUDAGuard guard(x.device());auto out=at::empty_like(x);auto st=at::cuda::getCurrentCUDAStream();
 int count=x.numel(),blocks=(count+511)/512;
 if(degree==0)map_exp<0><<<blocks,256,0,st>>>(x.data_ptr<float>(),out.data_ptr<float>(),count);
 else if(degree==3)map_exp<3><<<blocks,256,0,st>>>(x.data_ptr<float>(),out.data_ptr<float>(),count);
 else if(degree==4)map_exp<4><<<blocks,256,0,st>>>(x.data_ptr<float>(),out.data_ptr<float>(),count);
 else TORCH_CHECK(false,"unsupported degree");
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor probe_cost(at::Tensor anchor,int degree,int loops){
 c10::cuda::CUDAGuard guard(anchor.device());auto out=at::empty({264*128},anchor.options().dtype(at::kFloat));auto st=at::cuda::getCurrentCUDAStream();
 if(degree==0)register_probe<0><<<264,128,0,st>>>(out.data_ptr<float>(),loops);
 else if(degree==3)register_probe<3><<<264,128,0,st>>>(out.data_ptr<float>(),loops);
 else if(degree==4)register_probe<4><<<264,128,0,st>>>(out.data_ptr<float>(),loops);
 else TORCH_CHECK(false,"unsupported degree");
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
