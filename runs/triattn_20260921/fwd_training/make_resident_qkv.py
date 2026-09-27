"""Generate an isolated projection+attention prototype with resident BF16 QKV."""
from pathlib import Path
r = Path(__file__).resolve().parent
base = (r / 'cooperative_head2/fused.cu').read_text()
math = base[base.index('        uint32_t bp='):base.index('    warpgroup_wait<0>(); warpgroup_fence_operand(out);\n    float inv')]
math = math[:math.rindex('    }')]
math = math.replace('Config::', 'C::').replace('s.bias[stage].data()', 's.scratch.bias[c][stage].data()')
math = math.replace('s.v[stage][r].data()', 's.qkv[2][kt].data()')
math = math.replace('        float alpha[2], ls[2]={0.f,0.f};', '''        if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }
        float alpha[2], ls[2]={0.f,0.f};''')
epilogue = base[base.index('    warpgroup_wait<0>(); warpgroup_fence_operand(out);\n    float inv'):base.index('\nstd::vector<torch::Tensor> launch_forward')]
epilogue = epilogue[:epilogue.rindex('\n  }')]
epilogue = epilogue.replace('Config::', 'C::').replace('s.q[r].data()', 's.qkv[0][qt].data()').replace('128,r+1', '128,c+1')
header = base[:base.index('struct Config')]
s = header + r'''
template<int Capacity,int Consumers,int Stages> struct Config {
  using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using VT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
  using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_128>{}));
  using ZS=Shape<int32_t,_128,int32_t>; using ZStride=Stride<_128,_1,int64_t>;
  using WS=Shape<_128,_128>; using WStride=Stride<_128,_1>;
  using QS=Shape<int32_t,_32,_4,int32_t>; using QStride=Stride<int64_t,_1,_32,int64_t>;
  using BS=Shape<int32_t,int32_t,_4>; using BStride=Stride<int64_t,_1,int64_t>;
  using TZ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ZS{},ZStride{}),ZL{},Shape<_64,_128>{},_1{}));
  using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},WStride{}),WL{},Shape<_32,_128>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),BS{},BStride{}),BL{},Shape<_64,_64>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
  using Proj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_128>>()));
  using Score=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_32>>()));
  using PV=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  union alignas(1024) Scratch {
    struct { array_aligned<Element,8192,1024> z; array_aligned<Element,4096,1024> w; } proj;
    array_aligned<Element,4096,1024> bias[Consumers][Stages];
  };
  struct Shared {
    array_aligned<Element,2048,1024> qkv[3][Capacity/64];
    Scratch scratch;
    cutlass::arch::ClusterTransactionBarrier zfull,wfull,bfull[Consumers][Stages];
  };
  struct Params { TZ z; TW wq,wk,wv; TB bias; TO saveq,savek,savev,out; float* lse; int L; };
};
template<class T,class G,class S>
__device__ __forceinline__ void tma_load(T const& t,G const& g,S const& s,
    cutlass::arch::ClusterTransactionBarrier& bar) {
  auto c=t.get_slice(_0{});
  copy(t.with(reinterpret_cast<uint64_t&>(bar)),c.partition_S(g),c.partition_D(s));
}
template<int Capacity,int Consumers,int Stages>
__global__ __launch_bounds__(Consumers*128,1)
void qkv_attention_resident(CUTE_GRID_CONSTANT typename Config<Capacity,Consumers,Stages>::Params const p) {
  using C=Config<Capacity,Consumers,Stages>;
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x,c=tid/128,lane=tid%128,h=blockIdx.x,row=blockIdx.y,L=p.L,nt=L/64;
  if(tid==0) {
    s.zfull.init(1); s.wfull.init(1);
    for(int cc=0;cc<Consumers;++cc)for(int b=0;b<Stages;++b)s.bfull[cc][b].init(1);
    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    prefetch_tma_descriptor(p.wq.get_tma_descriptor());
    prefetch_tma_descriptor(p.wk.get_tma_descriptor());
    prefetch_tma_descriptor(p.wv.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  // Projection producer. Its scratch aliases later bias tiles, never live together.
  if(c==0) {
    typename C::Proj mma; auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
    auto acc=partition_fragment_C(mma,Shape<_64,_32>{});
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z.data()),typename C::ZL{});
    auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),typename C::WL{});
    auto za=mt.partition_fragment_A(zs); auto wb=mt.partition_fragment_B(ws);
    for(int which=0;which<3;++which) {
      auto const& wt=(which==0?p.wq:(which==1?p.wk:p.wv));
      auto const& save=(which==0?p.saveq:(which==1?p.savek:p.savev));
      if(lane==0) {
        s.wfull.arrive_and_expect_tx(4096*sizeof(Element));
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        auto tile=local_tile(wg,Shape<_32,_128>{},make_coord(h,0));
        tma_load(wt,tile,ws,s.wfull);
      }
      s.wfull.wait(which%2);
      for(int qt=0;qt<nt;++qt) {
        if(lane==0) {
          s.zfull.arrive_and_expect_tx(8192*sizeof(Element));
          auto tile=local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(qt,0));
          tma_load(p.z,tile,zs,s.zfull);
        }
        s.zfull.wait((which*nt+qt)%2);
        flash::gemm<true,0>(mma,za,wb,acc);
        uint32_t dst=cast_smem_ptr_to_uint(s.qkv[which][qt].data());
        #pragma unroll
        for(int x=0;x<size(acc);x+=2) {
          int mr=get<0>(coord(x)), nd=get<1>(coord(x));
          int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd))*2;
          store_pair(dst+off,acc(x),acc(x+1));
        }
        cutlass::arch::fence_view_async_shared();
        // Retire all Z readers and publish every producer's BF16 store.
        cutlass::arch::NamedBarrier::sync(128,0);
        if(lane==0) {
          auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
          auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
          auto src=make_tensor(make_smem_ptr(s.qkv[which][qt].data()),typename C::QL{});
          auto cp=save.get_slice(_0{});
          copy(save,cp.partition_S(src),cp.partition_D(tile));
          tma_store_arrive(); tma_store_wait<0>();
        }
      }
      cutlass::arch::NamedBarrier::sync(128,0);
    }
  }
  // QKV saves are complete. The projection scratch may become bias storage.
  __syncthreads();
  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  typename C::Score smma; typename C::PV pmma;
  auto st=smma.get_slice(lane); auto pt=pmma.get_slice(lane);
  auto sc=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
  auto oc=pt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
  for(int qt=c;qt<nt;qt+=Consumers) {
    auto load_bias=[&](int kt) {
      int stage=kt%Stages;
      s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));
      auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
      tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.scratch.bias[c][stage].data()),typename C::BL{}),s.bfull[c][stage]);
    };
    if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);
    auto score=partition_fragment_C(smma,Shape<_64,_64>{});
    auto out=partition_fragment_C(pmma,Shape<_64,_32>{}); clear(out);
    float m[2]={-INFINITY,-INFINITY},l[2]={1.f,1.f};
    auto sq=make_tensor(make_smem_ptr(s.qkv[0][qt].data()),typename C::QL{});
    auto qa=st.partition_fragment_A(sq);
    for(int kt=0;kt<nt;++kt) {
      int stage=kt%Stages;
      auto sk=make_tensor(make_smem_ptr(s.qkv[1][kt].data()),typename C::QL{});
      auto kb=st.partition_fragment_B(sk);
      flash::gemm<true,0>(smma,qa,kb,score);
      cutlass::arch::NamedBarrier::sync(128,c+1);
      if constexpr(Stages==2) {
        if(lane==0 && kt>0 && kt+1<nt)load_bias(kt+1);
      }
      s.bfull[c][stage].wait(((qt/Consumers)*(nt/Stages)+kt/Stages)%2);
      asm volatile("":::"memory");
''' + math.replace('C::BL{}','typename C::BL{}').replace('C::VT{}','typename C::VT{}').replace('flash::convert_layout_acc_Aregs<C::PV>', 'flash::convert_layout_acc_Aregs<typename C::PV>') + '\n    }\n' + epilogue.replace('C::QL{}','typename C::QL{}') + r'''
  }
}

template<int Capacity,int Consumers,int Stages>
void launch(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor b,
            torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,torch::Tensor lse) {
  using C=Config<Capacity,Consumers,Stages>; int L=z.size(1);
  auto zg=make_tensor(make_gmem_ptr((Element const*)z.data_ptr()),make_shape(L,_128{},L),typename C::ZStride{_128{},_1{},int64_t(L)*128});
  auto tz=make_tma_copy(SM90_TMA_LOAD{},zg,typename C::ZL{},Shape<_64,_128>{},_1{});
  auto weight=[&](torch::Tensor const& w) {
    auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),typename C::WS{},typename C::WStride{});
    return make_tma_copy(SM90_TMA_LOAD{},g,typename C::WL{},Shape<_32,_128>{},_1{});
  };
  auto bg=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),typename C::BStride{L,_1{},int64_t(L)*L});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},bg,typename C::BL{},Shape<_64,_64>{},_1{});
  auto save=[&](torch::Tensor const& t) {
    auto g=make_tensor(make_gmem_ptr((Element*)t.data_ptr()),make_shape(L,_32{},_4{},L),typename C::QStride{128,_1{},_32{},int64_t(L)*128});
    return make_tma_copy(SM90_TMA_STORE{},g,typename C::QL{},Shape<_64,_32>{},_1{});
  };
  typename C::Params p{tz,weight(wq),weight(wk),weight(wv),tb,save(q),save(k),save(v),save(out),lse.data_ptr<float>(),L};
  C10_CUDA_CHECK(cudaFuncSetAttribute(qkv_attention_resident<Capacity,Consumers,Stages>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  qkv_attention_resident<Capacity,Consumers,Stages><<<dim3(4,L),Consumers*128,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
}
std::vector<torch::Tensor> launch_forward(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor b) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type()==torch::kBFloat16 && z.is_contiguous() && z.dim()==4 && z.size(0)==1 && z.size(3)==128,"contiguous B1 L L C128 BF16 Z required");
  c10::cuda::CUDAGuard guard(z.device()); int L=z.size(1);
  TORCH_CHECK(z.size(2)==L && (L==64 || L==128 || L==256 || L==384 || L==768 || L==1024),"unsupported shape");
  for(auto const& w:{wq,wk,wv})TORCH_CHECK(w.device()==z.device() && w.scalar_type()==z.scalar_type() && w.is_contiguous() && w.sizes()==torch::IntArrayRef({128,128}),"invalid projection weight");
  TORCH_CHECK(b.device()==z.device() && b.scalar_type()==z.scalar_type() && b.is_contiguous() && b.sizes()==torch::IntArrayRef({1,4,L,L}),"invalid bias");
  auto q=torch::empty_like(z),k=torch::empty_like(z),v=torch::empty_like(z),o=torch::empty_like(z);
  auto lse=torch::empty({1,4,L,L},z.options().dtype(torch::kFloat32));
  if(L<=128)launch<128,2,2>(z,wq,wk,wv,b,q,k,v,o,lse);
  else if(L<=384)launch<384,2,2>(z,wq,wk,wv,b,q,k,v,o,lse);
  else if(L==768)launch<768,4,2>(z,wq,wk,wv,b,q,k,v,o,lse);
  else launch<1024,4,1>(z,wq,wk,wv,b,q,k,v,o,lse);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto view=[&](torch::Tensor const& x){return x.view({1,L,L,4,32}).permute({0,3,1,2,4});};
  return {view(o),lse,view(q),view(k),view(v)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&launch_forward);
  m.def("smem",[](){return sizeof(Config<1024,4,1>::Shared);});
}
'''
p = r / 'resident_qkv'
p.mkdir(exist_ok=True)
(p / 'fused.cu').write_text(s)
