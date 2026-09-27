from pathlib import Path
import json,os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
name='triattn_half2_exp_probe_v1'
coefficients=json.loads((HERE/'half2-exp-coefficients.json').read_text())
definitions=[]
for degree in (3,4):
 bits=coefficients[str(degree)]['packed_bits']
 definitions.append('template<> struct Coeff<%d>{static __device__ __forceinline__ uint32_t at(int n){\n switch(n){%s}return 0;}};'%(degree,''.join('case %d:return 0x%08xu;'%(i,b) for i,b in enumerate(bits))))
header='''#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
template<int D>struct Coeff;
'''+ '\n'.join(definitions)+'''
__device__ __forceinline__ float native_ex2(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
template<int D> __device__ __forceinline__ float2 half2_exp_pair(float x0,float x1){
 if constexpr(D==0)return make_float2(native_ex2(x0),native_ex2(x1));
 else{
  int n0=__float2int_rn(x0),n1=__float2int_rn(x1);
  float f0=x0-__int2float_rn(n0),f1=x1-__int2float_rn(n1);
  auto h=__floats2half2_rn(f0,f1);uint32_t r=reinterpret_cast<uint32_t const&>(h),y=Coeff<D>::at(D);
  #pragma unroll
  for(int k=D-1;k>=0;--k){uint32_t c=Coeff<D>::at(k);asm("fma.rn.f16x2 %0,%0,%1,%2;":"+r"(y):"r"(r),"r"(c));}
  uint32_t a=((y&0xffffu)<<13)+((uint32_t(n0)+112u)<<23);
  uint32_t b=((y>>16)<<13)+((uint32_t(n1)+112u)<<23);
  float v0=__uint_as_float(a),v1=__uint_as_float(b);
  if(!(x0>=-120.f && x0<=120.f))v0=native_ex2(x0);
  if(!(x1>=-120.f && x1<=120.f))v1=native_ex2(x1);
  return make_float2(v0,v1);
 }
}
'''
(HERE/'half2_exp_probe.cuh').write_text(header)
src=HERE/'half2_exp_probe.cu'
src.write_text('''#include <ATen/ATen.h>
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
''')
cpp=HERE/'half2_exp_probe.cpp'
cpp.write_text('''#include <torch/extension.h>
at::Tensor probe_exp(at::Tensor,int);
at::Tensor probe_cost(at::Tensor,int,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("map",&probe_exp);m.def("cost",&probe_cost);}
''')
directory=HERE/('build_'+name);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name=name,sources=[str(cpp),str(src)],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','--keep','--keep-dir='+str(directory),'-Xptxas=-v'],
 build_directory=str(directory),verbose=True)
print('BUILT',name,flush=True)
