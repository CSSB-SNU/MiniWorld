// Delta = sum_d O*dO without materializing FP32 expanded tensors in HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
template<int D> __global__ void wide_delta(const __nv_bfloat16* o,
        const __nv_bfloat16* dy,float* delta,int L) {
    int lane=threadIdx.x%32;
    int task=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(task>=4*L*L)return;
    int query=task%L, outer=(task/L)%L, head=task/(L*L);
    int64_t offset=(int64_t(outer)*L+query)*(4*D)+head*D;
    float value=0;
    #pragma unroll
    for(int d=lane;d<D;d+=32)
        value+=__bfloat162float(o[offset+d])*__bfloat162float(dy[offset+d]);
    #pragma unroll
    for(int shift=16;shift;shift/=2)value+=__shfl_xor_sync(0xffffffffu,value,shift);
    if(lane==0)delta[task]=value;
}
torch::Tensor delta(torch::Tensor out,torch::Tensor dy) {
    TORCH_CHECK(out.is_cuda() && out.dim()==5 && out.size(0)==1 && out.size(1)==4,"B1 H4 required");
    c10::cuda::CUDAGuard guard(out.device());
    int L=out.size(2),D=out.size(4);
    TORCH_CHECK((D==64 || D==128) && out.size(3)==L,"wide square shape required");
    for(auto const& x:{out,dy}) {
        TORCH_CHECK(x.device()==out.device() && x.sizes()==out.sizes() && x.scalar_type()==torch::kBFloat16,"O/dO metadata mismatch");
        TORCH_CHECK(x.stride(4)==1 && x.stride(1)==D && x.stride(3)==4*D && x.stride(2)==L*4*D,"projection layout required");
    }
    auto result=torch::empty({1,4,L,L},out.options().dtype(torch::kFloat32));
    auto stream=at::cuda::getCurrentCUDAStream();
    int blocks=(4*L*L+7)/8;
    if(D==64)wide_delta<64><<<blocks,256,0,stream>>>((__nv_bfloat16*)out.data_ptr(),(__nv_bfloat16*)dy.data_ptr(),result.data_ptr<float>(),L);
    else wide_delta<128><<<blocks,256,0,stream>>>((__nv_bfloat16*)out.data_ptr(),(__nv_bfloat16*)dy.data_ptr(),result.data_ptr<float>(),L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("delta",&delta);}
