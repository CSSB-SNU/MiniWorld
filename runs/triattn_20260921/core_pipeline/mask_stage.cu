#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
// One CTA owns 32 pair rows. Broadcast input is read once per CTA; all
// outputs have one writer, including batch extents and the OR-mask words.
__global__ void broadcast_mask_stage(bool const* mask,int64_t sb,int64_t sk,int N,int S,int W,
 int* words,int* keyany,uint8_t* kinds,int* end,int* start,int* rowstart,int* rowend,int* counts){
 __shared__ unsigned bits[36];
 __shared__ int first,last;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,b=blockIdx.y,row0=blockIdx.x*32;
 for(int wi=warp;wi<W;wi+=8){
  int key=wi*32+lane;
  bool live=key<S && mask[b*sb+int64_t(key)*sk];
  unsigned word=__ballot_sync(0xffffffffu,live);
  if(lane==0){bits[wi]=word;if(blockIdx.x==0)keyany[b*W+wi]=int(word);}
 }
 __syncthreads();
 if(warp==0){
  int lo=INT_MAX,hi=-1;
  for(int wi=lane;wi<W;wi+=32){unsigned word=bits[wi];if(word){lo=min(lo,wi);hi=max(hi,wi);}}
  lo=__reduce_min_sync(0xffffffffu,lo);hi=__reduce_max_sync(0xffffffffu,hi);
  if(lane==0){
   first=hi<0?0:lo;last=hi+1;
   if(blockIdx.x==0){start[b]=first;end[b]=last;if(hi<0)atomicAdd(counts+1,N);}
  }
 }
 __syncthreads();
 for(int n=tid;n<32*W;n+=256){int row=row0+n/W;if(row<N)words[(int64_t(b)*N+row)*W+n%W]=int(bits[n%W]);}
 if(tid<32 && row0+tid<N){int64_t row=int64_t(b)*N+row0+tid;kinds[row]=last?0:2;rowstart[row]=first;rowend[row]=last;}
}
std::vector<at::Tensor> stage(at::Tensor mask,at::Tensor counts){
 TORCH_CHECK(mask.is_cuda() && mask.scalar_type()==at::kBool && mask.dim()==5 && mask.size(2)==1 && mask.size(3)==1 && mask.stride(1)==0,"broadcast CUDA mask [B,N,1,1,S] required");
 TORCH_CHECK(mask.size(0)>0 && mask.size(1)>0 && mask.size(4)>0 && mask.size(4)<=1024,"positive B,N and 1<=S<=1024 required");
 TORCH_CHECK(counts.device()==mask.device() && counts.scalar_type()==at::kInt && counts.is_contiguous() && counts.numel()==2,"counts must be contiguous device int32[2]");
 c10::cuda::CUDAGuard guard(mask.device());int B=mask.size(0),N=mask.size(1),S=mask.size(4),W=4*((S+127)/128+1);
 auto opts=mask.options().dtype(at::kInt);
 auto words=at::empty({B,N,W},opts),keyany=at::empty({B,W},opts),kinds=at::empty({B,N},opts.dtype(at::kByte));
 auto end=at::empty({B},opts),start=at::empty({B},opts),rowstart=at::empty({B,N},opts),rowend=at::empty({B,N},opts);
 broadcast_mask_stage<<<dim3((N+31)/32,B),256,0,at::cuda::getCurrentCUDAStream()>>>(mask.data_ptr<bool>(),mask.stride(0),mask.stride(4),N,S,W,words.data_ptr<int>(),keyany.data_ptr<int>(),kinds.data_ptr<uint8_t>(),end.data_ptr<int>(),start.data_ptr<int>(),rowstart.data_ptr<int>(),rowend.data_ptr<int>(),counts.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {words,keyany,kinds,end,start,rowstart,rowend};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("stage",&stage);}
