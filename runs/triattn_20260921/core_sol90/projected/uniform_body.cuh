// Rare all-masked fallback: project rounded V into output, then average it.
// All CTAs return immediately for any-valid-key inputs.
__global__ __launch_bounds__(128,4) void project_uniform_v(CUTE_GRID_CONSTANT Params const p){
 if(*p.valid)return;
 extern __shared__ __align__(128) unsigned char storage[];
 auto& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x,h=blockIdx.y,i0=blockIdx.z*R,j=blockIdx.x*M;
 if(tid==0){
  s.xr.init(1);s.wr.init(1);cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if(tid==0){
  auto dst=make_tensor(make_smem_ptr(s.w),SWKV{});
  auto src=local_tile(p.wkv.get_tma_tensor(p.wkvs),Shape<_64,_128>{},make_coord(h,0));
  auto slice=p.wkv.get_slice(_0{});
  s.wr.arrive_and_expect_tx(64*128*sizeof(Element));
  copy(p.wkv.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
 }
 s.wr.wait(0);asm volatile("":::"memory");
 auto wv=local_tile(make_tensor(make_smem_ptr(s.w),SWKV{}),Shape<_32,_128>{},make_coord(1,0));
 #pragma unroll 1
 for(int r=0;r<R;++r){
  if(tid==0){
   auto dst=make_tensor(make_smem_ptr(s.x),SX{});
   auto src=local_tile(p.x.get_tma_tensor(p.xs)(_,_,i0+r),Shape<_128,_128>{},make_coord(j/M,0));
   auto slice=p.x.get_slice(_0{});
   s.xr.arrive_and_expect_tx(M*C*sizeof(Element));
   copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr)),slice.partition_S(src),slice.partition_D(dst));
  }
  s.xr.wait(r&1);asm volatile("":::"memory");
  // MQ selects M64/N32 and writes the q shared tile. The operand here is W_V.
  project<MQ>(s,wv,r,0,tid);project<MQ>(s,wv,r,1,tid);
  cutlass::arch::fence_view_async_shared();__syncthreads();
  if(tid==0){
   auto src=make_tensor(make_smem_ptr(s.q[r]),SO{});
   auto dst=local_tile(p.out.get_tma_tensor(p.os)(_,_,h,i0+r),Shape<_128,_32>{},make_coord(j/M,0));
   auto slice=p.out.get_slice(_0{});
   copy(p.out,slice.partition_S(src),slice.partition_D(dst));tma_store_arrive();
  }
  __syncthreads();
 }
 if(tid==0)tma_store_wait<0>();
}
}
void projected_uniform_cuda(at::Tensor x,at::Tensor wkv,at::Tensor out,at::Tensor valid){
 using namespace triattn_projected_uniform;
 int L=check_pair(x);
 check_bf16(wkv,x,256*128,"packed KV weight");check_bf16(out,x,int64_t(L)*L*128,"output");
 TORCH_CHECK(valid.device()==x.device() && valid.scalar_type()==at::kInt && valid.numel()==1,"invalid mask flag");
 c10::cuda::CUDAGuard guard(x.device());
 GX xs=make_shape(L,128,L);GW wkvs=make_shape(256,128);GO os=make_shape(L,32,4,L);
 auto tx=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),xs,DX{128,_1{},int64_t(L)*128}),SX{},Shape<_128,_128>{},_1{});
 auto twkv=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)wkv.data_ptr()),wkvs,DW{128,_1{}}),SWKV{},Shape<_64,_128>{},_1{});
 auto to=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)out.data_ptr()),os,DO{32,_1{},int64_t(L)*32,int64_t(4)*L*32}),SO{},Shape<_128,_32>{},_1{});
 Params p{tx,twkv,to,xs,wkvs,os,L,valid.data_ptr<int>()};
 C10_CUDA_CHECK(cudaFuncSetAttribute(project_uniform_v,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 project_uniform_v<<<dim3(L/128,4,L/2),128,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
