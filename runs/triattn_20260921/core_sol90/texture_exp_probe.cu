// Diagnostic only: texture interpolation error and mixed EX2/TEX throughput.
// This is not an attention kernel or a speedup qualification.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do { auto status=(call); if(status!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(status));std::exit(2);} } while(0)

__device__ __forceinline__ float native_exp(float x){float y;asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float texture_exp(cudaTextureObject_t tex,float x){
    if(x>127.f || isnan(x))return native_exp(x);
    float value=tex1D<float>(tex,fmaf(x,128.f,16128.5f));
    return x>=-126.f ? value : 0.f;
}
__global__ void accuracy(cudaTextureObject_t tex,float2* out,int n){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=n)return;
    float x=fmaf(float(t)/float(n-1),253.f,-126.f);
    out[t]=make_float2(native_exp(x),texture_exp(tex,x));
}
template<int Fraction> __global__ void throughput(cudaTextureObject_t tex,float* out){
    extern __shared__ float reserved[];
    if(threadIdx.x==0)reserved[0]=0.f;
    __syncthreads();
    float p[16];
    #pragma unroll
    for(int i=0;i<16;++i)p[i]=0.f;
    #pragma unroll 1
    for(int it=0;it<128;++it){
        #pragma unroll
        for(int i=0;i<16;++i){
            float x=fmaf(float((threadIdx.x*73+i*179+it*11)&1023),1.f/128.f,-70.f);
            float y;
            if constexpr(Fraction>0){if(i%4<Fraction)y=texture_exp(tex,x);else y=native_exp(x);}
            else y=native_exp(x);
            p[i]=fmaf(p[i],0.99f,y);
        }
    }
    float sum=0.f;
    #pragma unroll
    for(int i=0;i<16;++i)sum+=p[i];
    out[blockIdx.x*blockDim.x+threadIdx.x]=sum+reserved[0];
}
template<int F> void launch(cudaTextureObject_t tex,float* out){throughput<F><<<132,384,190*1024>>>(tex,out);}
template<int F> void measure(cudaTextureObject_t tex,float* out){
    CHECK(cudaFuncSetAttribute(throughput<F>,cudaFuncAttributeMaxDynamicSharedMemorySize,190*1024));
    int blocks=0;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,throughput<F>,384,190*1024));
    if(blocks!=1){std::fprintf(stderr,"expected one CTA per SM, got %d\n",blocks);std::exit(3);}
    for(int i=0;i<5;++i)launch<F>(tex,out);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for(int r=0;r<5;++r){
        CHECK(cudaEventRecord(start));for(int i=0;i<10;++i)launch<F>(tex,out);
        CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));float ms;
        CHECK(cudaEventElapsedTime(&ms,start,stop));samples.push_back(ms*100.f);
    }
    std::sort(samples.begin(),samples.end());
    std::printf("MICRO fraction=%d/4 us=%.6f\n",F,samples[2]);
    CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));
}
int main(){
    constexpr int width=253*128+1,n=1<<20;
    std::vector<float> table(width);
    for(int i=0;i<width;++i)table[i]=float(std::exp2(double(i)/128.-126.));
    cudaArray_t array;auto channel=cudaCreateChannelDesc<float>();
    CHECK(cudaMallocArray(&array,&channel,width));
    CHECK(cudaMemcpy2DToArray(array,0,0,table.data(),width*sizeof(float),width*sizeof(float),1,cudaMemcpyHostToDevice));
    cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
    cudaTextureDesc desc{};desc.addressMode[0]=cudaAddressModeClamp;desc.filterMode=cudaFilterModeLinear;desc.readMode=cudaReadModeElementType;
    cudaTextureObject_t tex;CHECK(cudaCreateTextureObject(&tex,&resource,&desc,nullptr));
    float2* output;CHECK(cudaMalloc(&output,n*sizeof(float2)));
    accuracy<<<(n+255)/256,256>>>(tex,output,n);CHECK(cudaDeviceSynchronize());
    std::vector<float2> pairs(n);CHECK(cudaMemcpy(pairs.data(),output,n*sizeof(float2),cudaMemcpyDeviceToHost));
    double maximum=0.,rms=0.,ordinary_max=0.;int bad=0;
    for(int i=0;i<n;++i){
        double e=double(pairs[i].y)/pairs[i].x-1.;float x=float(i)/float(n-1)*253.f-126.f;
        if(!std::isfinite(e)){++bad;continue;}
        maximum=std::max(maximum,std::abs(e));rms+=e*e;
        if(x>=-80.f && x<=-40.f)ordinary_max=std::max(ordinary_max,std::abs(e));
    }
    std::printf("ERROR max=%.12g rms=%.12g ordinary_max=%.12g bad=%d\n",maximum,std::sqrt(rms/n),ordinary_max,bad);
    measure<0>(tex,reinterpret_cast<float*>(output));measure<1>(tex,reinterpret_cast<float*>(output));
    measure<2>(tex,reinterpret_cast<float*>(output));measure<4>(tex,reinterpret_cast<float*>(output));
    CHECK(cudaFree(output));CHECK(cudaDestroyTextureObject(tex));CHECK(cudaFreeArray(array));
}
