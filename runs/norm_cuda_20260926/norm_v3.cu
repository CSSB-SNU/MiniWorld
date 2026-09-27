// Generic last-axis normalization. No atomics: bounded FP32/FP64 partials for
// affine gradients, then a deterministic column reduction. Warp owns a row.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>

template<class A> __device__ A sumwarp(A v) {
  for (int d=16; d; d>>=1) v += __shfl_down_sync(0xffffffff, v, d);
  return __shfl_sync(0xffffffff, v, 0);
}
template<class T, class A, bool RMS>
__global__ void forward_norm(const T* x, const A* w, const A* b, T* y,
                            A* mean, A* inv, int64_t M, int D, A eps) {
  int lane=threadIdx.x%32;
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  for(;row<M;row+=int64_t(gridDim.x)*(blockDim.x/32)) {
  A mu=0;
  if(!RMS) { for(int c=lane;c<D;c+=32) mu+=A(x[row*D+c]); mu=sumwarp(mu)/D; }
  A var=0;
  for(int c=lane;c<D;c+=32) {A z=A(x[row*D+c])-mu;var+=z*z;}
  A r=A(1)/sqrt(sumwarp(var)/D+eps);
  if(lane==0) {mean[row]=mu;inv[row]=r;}
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    y[row*D+c]=T(z*(w?w[c]:A(1))+(b?b[c]:A(0)));
  }
  }
}
// Width slices of 32 are revisited after the two row reductions. A warp scans
// ROWS rows; affine partials are assigned to a second column kernel to avoid
// storing D-sized register arrays (arbitrary D and predictable register usage).
template<class T,class A,bool RMS>
__global__ void backward_dx(const T* x,const T* dy,const A* w,const A* mean,
                           const A* inv,T* dx,int64_t M,int D) {
  int lane=threadIdx.x%32;
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  if(row>=M) return;
  A mu=mean[row],r=inv[row],s1=0,s2=0;
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    A g=A(dy[row*D+c])*(w?w[c]:A(1));
    s1+=g;s2+=g*z;
  }
  if(!RMS) s1=sumwarp(s1)/D;
  s2=sumwarp(s2)/D;
  for(int c=lane;c<D;c+=32) {
    A z=(A(x[row*D+c])-mu)*r;
    A g=A(dy[row*D+c])*(w?w[c]:A(1));
    dx[row*D+c]=T((g-(RMS?A(0):s1)-z*s2)*r);
  }
}
template<class T,class A>
__global__ void affine_partial(const T* x,const T* dy,const A* mean,const A* inv,
                              A* pw,A* pb,int64_t M,int D,int rows) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=D) return;
  int64_t first=int64_t(blockIdx.y)*rows,last=min(first+rows,M);
  A sw=0,sb=0;
  for(int64_t row=first;row<last;++row) {
    A g=A(dy[row*D+c]);
    if(pw) sw+=g*(A(x[row*D+c])-mean[row])*inv[row];
    if(pb) sb+=g;
  }
  if(pw) pw[int64_t(blockIdx.y)*D+c]=sw;
  if(pb) pb[int64_t(blockIdx.y)*D+c]=sb;
}

// Fused dX + dgamma/dbeta partials. Values loaded once into registers; each warp
// owns ROWS consecutive rows, avoiding atomics and a second x/dy HBM pass.
template<class T,class A,bool RMS,int C>
__global__ void backward_fused(const T* x,const T* dy,const A* w,const A* mean,
 const A* inv,T* dx,A* pw,A* pb,int64_t M,int D,int rows,int P) {
  int lane=threadIdx.x%32;
  int group=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(group>=P)return;
  A sw[C]={},sb[C]={};
  for(int64_t row=group;row<M;row+=P) {
    A z[C],g[C],s1=0,s2=0,mu=mean[row],r=inv[row];
    #pragma unroll
    for(int k=0;k<C;++k) {
      int c=lane+32*k;A v=c<D?A(dy[row*D+c]):A(0);
      z[k]=c<D?(A(x[row*D+c])-mu)*r:A(0);
      g[k]=v*(w&&c<D?w[c]:A(1));
      s1+=g[k];s2+=g[k]*z[k];
      if(pw)sw[k]+=v*z[k];if(pb)sb[k]+=v;
    }
    if(!RMS)s1=sumwarp(s1)/D;
    s2=sumwarp(s2)/D;
    #pragma unroll
    for(int k=0;k<C;++k) {
      int c=lane+32*k;if(c<D)dx[row*D+c]=T((g[k]-(RMS?A(0):s1)-z[k]*s2)*r);
    }
  }
  #pragma unroll
  for(int k=0;k<C;++k) {
    int c=lane+32*k;
    if(c<D) {if(pw)pw[int64_t(group)*D+c]=sw[k];if(pb)pb[int64_t(group)*D+c]=sb[k];}
  }
}
template<class A>
__global__ void affine_finish(const A* pw,const A* pb,A* dw,A* db,int P,int D) {
  int c=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  A sw=0,sb=0;
  for(int p=threadIdx.x;p<P;p+=blockDim.x) {if(pw)sw+=pw[int64_t(p)*D+c];if(pb)sb+=pb[int64_t(p)*D+c];}
  sw=sumwarp(sw);sb=sumwarp(sb);
  __shared__ A ws[8],bs[8];
  if(lane==0){ws[warp]=sw;bs[warp]=sb;}__syncthreads();
  if(warp==0) {
    sw=sumwarp(lane<8?ws[lane]:A(0));sb=sumwarp(lane<8?bs[lane]:A(0));
    if(lane==0){if(dw)dw[c]=sw;if(db)db[c]=sb;}
  }
}
void validate(torch::Tensor x,torch::Tensor w,torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim()>=1,"contiguous CUDA input required");
  TORCH_CHECK(x.size(-1)>0 && x.size(-1)<=INT_MAX,"invalid normalized width");
  auto acc=x.scalar_type()==at::kDouble?at::kDouble:at::kFloat;
  for(auto p:{w,b}) if(p.defined()) {
    TORCH_CHECK(p.device()==x.device() && p.scalar_type()==acc && p.is_contiguous() && p.dim()==1 && p.numel()==x.size(-1),"invalid affine parameter");
  }
}
std::vector<torch::Tensor> norm_forward(torch::Tensor x,c10::optional<torch::Tensor> weight,
 c10::optional<torch::Tensor> bias,double eps,bool rms,int threads) {
  torch::Tensor w=weight.value_or(torch::Tensor()),b=bias.value_or(torch::Tensor());validate(x,w,b);
  TORCH_CHECK(threads==128||threads==256,"invalid block size");
  c10::cuda::CUDAGuard guard(x.device());
  auto opt=x.options().dtype(x.scalar_type()==at::kDouble?at::kDouble:at::kFloat);
  int D=x.size(-1);int64_t M=x.numel()/D;
  auto y=torch::empty_like(x),mean=torch::empty({M},opt),inv=torch::empty({M},opt);
  if(M) AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"norm_forward",[&]{
    using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;
    auto stream=at::cuda::getCurrentCUDAStream();
    int64_t grid=std::min<int64_t>((M+threads/32-1)/(threads/32),8*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    if(rms) forward_norm<scalar_t,A,true><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps));
    else forward_norm<scalar_t,A,false><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,b.defined()?b.data_ptr<A>():nullptr,y.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),M,D,A(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,mean,inv};
}
#define LAUNCH(C,R) backward_fused<scalar_t,A,R,C><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,rows,P)
#define WIDTH(C) if(rms){LAUNCH(C,true);}else{LAUNCH(C,false);}
std::vector<torch::Tensor> norm_backward(torch::Tensor x,torch::Tensor dy,
 c10::optional<torch::Tensor> weight,torch::Tensor mean,torch::Tensor inv,
 bool has_bias,bool rms,int threads,int rows) {
  torch::Tensor w=weight.value_or(torch::Tensor());validate(x,w,torch::Tensor());
  TORCH_CHECK(dy.sizes()==x.sizes() && dy.scalar_type()==x.scalar_type() && dy.device()==x.device() && dy.is_contiguous(),"invalid dy");
  TORCH_CHECK((threads==128||threads==256)&&rows>=1&&rows<=4096,"invalid launch configuration");
  c10::cuda::CUDAGuard guard(x.device());
  int D=x.size(-1);int64_t M=x.numel()/D;
  int64_t max_groups=std::max<int64_t>(1,std::min<int64_t>(65535,(32*1024*1024)/(2*D*mean.element_size())));
  rows=std::max<int64_t>(rows,(M+max_groups-1)/max_groups);
  int P=(M+rows-1)/rows;
  auto dx=torch::empty_like(x),dw=torch::empty({w.defined()?D:0},mean.options()),db=torch::empty({has_bias?D:0},mean.options());
  auto pw=torch::empty({w.defined()?P:0,D},mean.options()),pb=torch::empty({has_bias?P:0,D},mean.options());
  TORCH_CHECK(mean.device()==x.device()&&inv.device()==x.device()&&mean.numel()==M&&inv.numel()==M&&mean.is_contiguous()&&inv.is_contiguous(),"invalid statistics");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),"norm_backward",[&]{
    using A=std::conditional_t<std::is_same_v<scalar_t,double>,double,float>;
    auto stream=at::cuda::getCurrentCUDAStream();
    if(M && D<=1024 && (w.defined()||has_bias)) {
      int grid=(P+threads/32-1)/(threads/32);
      if(D<=128){WIDTH(4)}else if(D<=256){WIDTH(8)}else if(D<=512){WIDTH(16)}else{WIDTH(32)}
    }else if(M) {
      int64_t grid=(M+threads/32-1)/(threads/32);
      if(rms) backward_dx<scalar_t,A,true><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),M,D);
      else backward_dx<scalar_t,A,false><<<grid,threads,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),w.defined()?w.data_ptr<A>():nullptr,mean.data_ptr<A>(),inv.data_ptr<A>(),dx.data_ptr<scalar_t>(),M,D);
    }
    if(w.defined()||has_bias) {
      if(M && D>1024) affine_partial<scalar_t,A><<<dim3((D+127)/128,P),128,0,stream>>>(x.data_ptr<scalar_t>(),dy.data_ptr<scalar_t>(),mean.data_ptr<A>(),inv.data_ptr<A>(),w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,M,D,rows);
      affine_finish<A><<<D,256,0,stream>>>(w.defined()?pw.data_ptr<A>():nullptr,has_bias?pb.data_ptr<A>():nullptr,w.defined()?dw.data_ptr<A>():nullptr,has_bias?db.data_ptr<A>():nullptr,P,D);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw,db};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&norm_forward);m.def("backward",&norm_backward);}
