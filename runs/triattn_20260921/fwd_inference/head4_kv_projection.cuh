// Four D32 heads are one dense N128 projection. Z is loaded once for K and V.
struct Head4KV {
  using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
  using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_128,_128>{}));
  using OL=ZL;
  using GS=Shape<int32_t,_128>; using Stride2=Stride<_128,_1>;
  using WS=Shape<_128,_128>;
  using TZ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},Stride2{}),ZL{},Shape<_64,_128>{},_1{}));
  using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},Stride2{}),WL{},Shape<_128,_128>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),GS{},Stride2{}),OL{},Shape<_64,_128>{},_1{}));
  using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_128,_128>>()));
  struct Shared {
    array_aligned<Element,8192,1024> z,out;
    array_aligned<Element,16384,1024> w;
    cutlass::arch::ClusterTransactionBarrier full;
  };
  struct Params { TZ z; TW wk,wv; TO key,value; int rows; };
};
__global__ __launch_bounds__(128,2)
void head4_kv_projection(CUTE_GRID_CONSTANT Head4KV::Params const p) {
  using C=Head4KV;
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<C::Shared*>(storage);
  int lane=threadIdx.x;
  if(lane==0) { s.full.init(1);cutlass::arch::fence_barrier_init(); }
  __syncthreads();
  auto zs=make_tensor(make_smem_ptr(s.z.data()),C::ZL{});
  auto ws=make_tensor(make_smem_ptr(s.w.data()),C::WL{});
  C::MMA mma;auto mt=mma.get_slice(lane);
  auto za=mt.partition_fragment_A(zs);auto wb=mt.partition_fragment_B(ws);
  auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_128>{}));
  auto acc=partition_fragment_C(mma,Shape<_64,_128>{});
  for(int which=0;which<2;++which) {
    if(lane==0) {
      s.full.arrive_and_expect_tx((16384+(which==0?8192:0))*sizeof(Element));
      if(which==0) {
        auto g=p.z.get_tma_tensor(make_shape(p.rows,_128{}));
        auto tile=local_tile(g,Shape<_64,_128>{},make_coord(blockIdx.x,0));
        tma_load(p.z,tile,zs,s.full);
      }
      auto const& t=which==0?p.wk:p.wv;
      auto g=t.get_tma_tensor(make_shape(_128{},_128{}));
      tma_load(t,g,ws,s.full);
    }
    s.full.wait(which);
    flash::gemm<true,0>(mma,za,wb,acc);
    #pragma unroll
    for(int x=0;x<size(acc);x+=2) {
      int off=as_position_independent_swizzle_layout(C::OL{})(coord(x))*2;
      store_pair(cast_smem_ptr_to_uint(s.out.data())+off,acc(x),acc(x+1));
    }
    cutlass::arch::fence_view_async_shared();__syncthreads();
    if(lane==0) {
      auto const& t=which==0?p.key:p.value;
      auto g=t.get_tma_tensor(make_shape(p.rows,_128{}));
      auto tile=local_tile(g,Shape<_64,_128>{},make_coord(blockIdx.x,0));
      auto out=make_tensor(make_smem_ptr(s.out.data()),C::OL{});auto ts=t.get_slice(_0{});
      copy(t,ts.partition_S(out),ts.partition_D(tile));tma_store_arrive();tma_store_wait<0>();
    }
    __syncthreads();
  }
}
void project_head4_kv(torch::Tensor z,torch::Tensor wk,torch::Tensor wv,torch::Tensor key,torch::Tensor value) {
  using C=Head4KV;int rows=z.numel()/128;
  auto zg=make_tensor(make_gmem_ptr((Element const*)z.data_ptr()),make_shape(rows,_128{}),C::Stride2{});
  auto tz=make_tma_copy(SM90_TMA_LOAD{},zg,C::ZL{},Shape<_64,_128>{},_1{});
  auto weight=[&](torch::Tensor const& w) {
    auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),C::WS{},C::Stride2{});
    return make_tma_copy(SM90_TMA_LOAD{},g,C::WL{},Shape<_128,_128>{},_1{});
  };
  auto output=[&](torch::Tensor const& o) {
    auto g=make_tensor(make_gmem_ptr((Element*)o.data_ptr()),make_shape(rows,_128{}),C::Stride2{});
    return make_tma_copy(SM90_TMA_STORE{},g,C::OL{},Shape<_64,_128>{},_1{});
  };
  C::Params p{tz,weight(wk),weight(wv),output(key),output(value),rows};
  C10_CUDA_CHECK(cudaFuncSetAttribute(head4_kv_projection,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(C::Shared)));
  head4_kv_projection<<<rows/64,128,sizeof(C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
