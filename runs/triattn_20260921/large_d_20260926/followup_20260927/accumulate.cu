// Four large products reuse one output through cuBLAS beta=1; fuse narrow bias.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
__global__ void projection_bias_accumulate(__nv_bfloat16* dx,const __nv_bfloat16* db,const __nv_bfloat16* wb,int rows,int C){
  int64_t n=int64_t(rows)*C;
  for(int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<n;idx+=int64_t(blockDim.x)*gridDim.x){
    int row=idx/C,col=idx%C;float x=__bfloat162float(dx[idx]);
    #pragma unroll
    for(int h=0;h<4;++h)x=fmaf(__bfloat162float(db[row*4+h]),__bfloat162float(wb[h*C+col]),x);
    dx[idx]=__float2bfloat16_rn(x);
  }
}
torch::Tensor accumulate(torch::Tensor dx,torch::Tensor db,torch::Tensor wb){
  TORCH_CHECK(dx.is_cuda() && dx.dim()==2 && (dx.size(1)==256 || dx.size(1)==512),"wide dx required");
  c10::cuda::CUDAGuard guard(dx.device());int rows=dx.size(0),C=dx.size(1);
  for(auto x:{dx,db,wb})TORCH_CHECK(x.device()==dx.device() && x.is_contiguous() && x.scalar_type()==torch::kBFloat16,"contiguous BF16 required");
  TORCH_CHECK(db.numel()==int64_t(rows)*4 && wb.sizes()==torch::IntArrayRef({4,C}),"shape mismatch");
  int blocks=std::min<int64_t>(4096,(int64_t(rows)*C+255)/256);
  projection_bias_accumulate<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)dx.data_ptr(),(__nv_bfloat16*)db.data_ptr(),(__nv_bfloat16*)wb.data_ptr(),rows,C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return dx;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("accumulate",&accumulate);m.def("smem",[](){return 0;});}
