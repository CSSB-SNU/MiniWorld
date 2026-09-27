#pragma once
#include <ATen/ATen.h>
inline void check_bf16(at::Tensor const& t,at::Tensor const& z,int64_t count,char const* name,bool contiguous=true){
 TORCH_CHECK(t.is_cuda() && t.device()==z.device() && t.scalar_type()==at::kBFloat16 && t.numel()==count && (!contiguous || t.is_contiguous()),name," has an unsupported device, dtype, size or stride");
}
inline int check_pair(at::Tensor const& z){
 TORCH_CHECK(z.is_cuda() && z.scalar_type()==at::kBFloat16 && z.dim()>=3 && z.is_contiguous(),"contiguous CUDA bf16 pair required");
 int64_t L=z.size(-2);
 TORCH_CHECK((L==384 || L==768 || L==1024) && z.size(-3)==L && z.size(-1)==128 && z.numel()==L*L*128,"supported shape is one square C128 pair at L384/768/1024");
 return int(L);
}
