void projected_uniform_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor);
at::Tensor projected_forward_cuda(at::Tensor x,at::Tensor w,at::Tensor wkv,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace triattn_projected_fused;
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.dim()==3 && x.size(0)==L && x.size(1)==L && x.size(2)==128,"L768 normalized BF16 features required");
 for(auto const& t:{w,wkv})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"invalid projection weights");
 TORCH_CHECK(w.numel()==512*128 && wkv.numel()==256*128,"invalid weight shape");
 TORCH_CHECK(bias.device()==x.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==4*L*L,"invalid bias");
 TORCH_CHECK(float(scale)==0x1.6a09e6p-3f,"standard D32 scale required");
 bool const* mp=nullptr;int64_t ms=1;
 if(mask){auto m=*mask;TORCH_CHECK(m.device()==x.device() && m.scalar_type()==at::kBool && m.dim()==5 && m.size(0)==1 && m.size(1)==L && m.size(2)==1 && m.size(3)==1 && m.size(4)==L && m.stride(1)==0,"row-broadcast mask required");mp=m.data_ptr<bool>();ms=m.stride(4);}
 c10::cuda::CUDAGuard guard(x.device());
 constexpr int ct=6*(L/Rows)*4;
 auto output=at::empty({L,4,L,32},x.options()),prepared=at::empty_like(bias);
 auto fix=at::empty({ct},x.options().dtype(at::kInt)),valid=at::empty({1},fix.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 prepare<<<dim3(L/LN,L/M,4),256,0,stream>>>(bias.data_ptr<float>(),mp,ms,float(1./scale),L,prepared.data_ptr<float>(),valid.data_ptr<int>(),fix.data_ptr<int>());
 auto tx=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,128,L),DX{128,_1{},int64_t(L)*128}),SX{},Shape<_128,_128>{},_1{});
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),make_shape(512,128),DW{128,_1{}}),SWQ{},Shape<_32,_128>{},_1{});
 auto twkv=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)wkv.data_ptr()),make_shape(256,128),DW{128,_1{}}),SWKV{},Shape<_64,_128>{},_1{});
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(prepared.data_ptr<float>()),make_shape(256,8,48,6,4),DB{_1{},_256{},64*N,int64_t(L/N)*M*N,int64_t(L/M)*(L/N)*M*N}),SB{},Shape<_256,_8>{},_1{});
 Params p{tx,tw,twkv,tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale)};
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);
 projected_uniform_cuda(x,wkv,output,valid);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)output.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
