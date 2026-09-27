// Compare actual H100 F16/F32 EX2 throughput. Instruction count is not a rate.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(1);}}
template<bool Half,int Count> __global__ __launch_bounds__(256,4)
void exp_rate(float* output,int loops){
 float f[Count];unsigned h[Count/2];
 #pragma unroll
 for(int i=0;i<Count;++i)f[i]=-0.3f-float((threadIdx.x+i)%16)*0.001f;
 if constexpr(Half){
  #pragma unroll
  for(int i=0;i<Count/2;++i){auto v=__floats2half2_rn(f[2*i],f[2*i+1]);h[i]=reinterpret_cast<unsigned const&>(v);}
 }
 #pragma unroll 1
 for(int j=0;j<loops;++j){
  if constexpr(Half){
   #pragma unroll
   for(int i=0;i<Count/2;++i){
    asm volatile("ex2.approx.f16x2 %0,%0;":"+r"(h[i]));
    asm volatile("mul.rn.f16x2 %0,%0,%1;":"+r"(h[i]):"r"(0xb800b800u));
   }
  }else{
   #pragma unroll
   for(int i=0;i<Count;++i){
    asm volatile("ex2.approx.ftz.f32 %0,%0;":"+f"(f[i]));
    asm volatile("mul.rn.f32 %0,%0,0fBF000000;":"+f"(f[i]));
   }
  }
 }
 float sum=0;
 if constexpr(Half){
  #pragma unroll
  for(int i=0;i<Count/2;++i){auto x=__half22float2(reinterpret_cast<__half2 const&>(h[i]));sum+=x.x+x.y;}
 }else{
  #pragma unroll
  for(int i=0;i<Count;++i)sum+=f[i];
 }
 output[blockIdx.x*256+threadIdx.x]=sum;
}
template<int Count> void compare(float* p,int blocks){
 float records[2][8];cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
 constexpr int loops=4096;
 for(int rd=0;rd<8;++rd)for(int order=0;order<2;++order){
  int half=(order+rd)%2;
  for(int warm=0;warm<3;++warm){if(half)exp_rate<true,Count><<<blocks,256>>>(p,loops);else exp_rate<false,Count><<<blocks,256>>>(p,loops);}
  ck(cudaEventRecord(start));
  if(half)exp_rate<true,Count><<<blocks,256>>>(p,loops);else exp_rate<false,Count><<<blocks,256>>>(p,loops);
  ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));ck(cudaGetLastError());ck(cudaEventElapsedTime(&records[half][rd],start,end));
 }
 for(int half=0;half<2;++half){
  printf("{\"half\":%s,\"independent_logits\":%d,\"blocks\":%d,\"loops\":%d,\"ms\":[",half?"true":"false",Count,blocks,loops);
  for(int rd=0;rd<8;++rd)printf("%s%.6f",rd?",":"",records[half][rd]);puts("]}");
 }
 ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
}
int main(){
 cudaDeviceProp dev;ck(cudaGetDeviceProperties(&dev,0));printf("DEVICE %s SM%d\n",dev.name,dev.multiProcessorCount);
 float* out;int blocks=dev.multiProcessorCount*4;ck(cudaMalloc(&out,blocks*256*sizeof(float)));
 compare<8>(out,blocks);compare<16>(out,blocks);compare<32>(out,blocks);ck(cudaFree(out));
}
