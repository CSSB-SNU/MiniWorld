// Diagnostic instruction attribution only. This is not an attention kernel.
#include <cuda_runtime.h>
#include <cstdio>
template<int Kind> __global__ void xu_probe(unsigned* out){
 unsigned x[16];
 #pragma unroll
 for(int i=0;i<16;i++)x[i]=0x3f000000u+threadIdx.x+i;
 #pragma unroll 1
 for(int it=0;it<256;it++){
  #pragma unroll
  for(int i=0;i<16;i++){
   if constexpr(Kind==0){float f=__uint_as_float(x[i]);asm volatile("ex2.approx.ftz.f32 %0,%0;":"+f"(f));x[i]=__float_as_uint(f);}
   if constexpr(Kind==1){unsigned y;asm volatile("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(y):"f"(__uint_as_float(x[i])),"f"(__uint_as_float(x[(i+1)%16])));x[i]=y;}
   if constexpr(Kind==2)asm volatile("prmt.b32 %0,%0,%1,0x7632;":"+r"(x[i]):"r"(x[(i+1)%16]));
   if constexpr(Kind==3)asm volatile("shr.u32 %0,%0,16;":"+r"(x[i]));
   if constexpr(Kind==4)asm volatile("lop3.b32 %0,%0,%1,0xffff0000,0xe2;":"+r"(x[i]):"r"(x[(i+1)%16]));
   if constexpr(Kind==5)asm volatile("add.u32 %0,%0,0x8000;":"+r"(x[i]));
   if constexpr(Kind==6)asm volatile("vadd.u32.u32.u32.min %0,%0,0x8000,0x7fc08000;":"+r"(x[i]));
  }
 }
 unsigned sum=0;
 #pragma unroll
 for(int i=0;i<16;i++)sum^=x[i];
 out[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
template<int K> void run(unsigned* out){xu_probe<K><<<132,256>>>(out);}
int main(){
 unsigned* out=nullptr;
 if(cudaMalloc(&out,132*256*sizeof(unsigned))!=cudaSuccess)return 1;
 run<0>(out);run<1>(out);run<2>(out);run<3>(out);run<4>(out);run<5>(out);run<6>(out);
 auto status=cudaDeviceSynchronize();
 if(status!=cudaSuccess)std::fprintf(stderr,"%s\n",cudaGetErrorString(status));
 cudaFree(out);return status==cudaSuccess?0:2;
}
