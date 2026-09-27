#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
template<int D>struct Coeff;
template<> struct Coeff<3>{static __device__ __forceinline__ uint32_t at(int n){
 switch(n){case 0:return 0x3c003c00u;case 1:return 0x398c398cu;case 2:return 0x33c133c1u;case 3:return 0x2b0a2b0au;}return 0;}};
template<> struct Coeff<4>{static __device__ __forceinline__ uint32_t at(int n){
 switch(n){case 0:return 0x3bff3bffu;case 1:return 0x398b398bu;case 2:return 0x33b133b1u;case 3:return 0x2b262b26u;case 4:return 0x20e320e3u;}return 0;}};
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
