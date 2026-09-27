// Row-group dK/dV + shared-bias gradient. Native CUDA/TMA/WGMMA, B1 H4 D32.
// One TMA producer, two consumer warpgroups; each WG owns R/2 outer rows.
// dK/dV are final outputs. Only one FP32 bias partial per R rows reaches HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr float SCALE=0.1767766952966369f;
constexpr float LOG2E=1.44269504f;

template<int R> struct Config {
  static constexpr int NWG=2, RP=R/NWG;
  using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>{}));
  using SL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using ShapeQ=Shape<int32_t,_32,_4,int32_t>;
  using StrideQ=Stride<int64_t,_1,_32,int64_t>;
  using ShapeB=Shape<int32_t,int32_t,_4>;
  using StrideB=Stride<int64_t,_1,int64_t>;
  using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ShapeQ{},StrideQ{}),QL{},Shape<_64,_32>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ShapeB{},StrideB{}),SL{},Shape<_64,_64>{},_1{}));
  using ScoreMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_32>>()));
  using GradMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  struct Shared {
    array_aligned<Element,2048,1024> k[R],v[R];
    array_aligned<Element,2048,1024> q[NWG][2],dout[NWG][2];
    array_aligned<Element,4096,1024> bias[2],prob[NWG],ds[NWG];
    array_aligned<float,4096,128> db[NWG]; // fragment order, one element per thread/register
    cutlass::arch::ClusterTransactionBarrier kv_full;
    cutlass::arch::ClusterTransactionBarrier q_full[NWG][2],b_full[2];
    cutlass::arch::ClusterBarrier q_empty[NWG][2],b_empty[2];
  };
  struct Params {
    TQ q,k,v,dout; TB bias;
    float const *lse,*delta;
    Element *dk,*dv;
    float *db;
    int L;
  };
};

template<class TMA,class GT,class ST>
__device__ __forceinline__ void tma_copy(TMA const& tma,GT const& src,ST const& dst,cutlass::arch::ClusterTransactionBarrier& full) {
  auto c=tma.get_slice(_0{});
  copy(tma.with(reinterpret_cast<uint64_t&>(full)),c.partition_S(src),c.partition_D(dst));
}

template<int R> __global__ __launch_bounds__(384,1)
void grouped_dkdv(CUTE_GRID_CONSTANT typename Config<R>::Params const p) {
  using C=Config<R>;
  extern __shared__ char smem[];
  auto& s=*reinterpret_cast<typename C::Shared*>(smem);
  int tid=threadIdx.x, wg=tid/128-1, lane=tid%128;
  int h=blockIdx.z, group=blockIdx.y, kt=blockIdx.x;
  int L=p.L, nt=L/64;
  if(tid==0) {
    s.kv_full.init(1);
    for(int st=0;st<2;++st) {
      s.b_full[st].init(1);s.b_empty[st].init(2);
      for(int w=0;w<2;++w) {s.q_full[w][st].init(1);s.q_empty[w][st].init(1);}
    }
    prefetch_tma_descriptor(p.q.get_tma_descriptor());prefetch_tma_descriptor(p.k.get_tma_descriptor());
    prefetch_tma_descriptor(p.v.get_tma_descriptor());prefetch_tma_descriptor(p.dout.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if(tid==0) {
      auto shape=make_shape(L,_32{},_4{},L);
      auto mk=p.k.get_tma_tensor(shape);auto mv=p.v.get_tma_tensor(shape);
      s.kv_full.arrive_and_expect_tx(R*2*2048*sizeof(Element));
      for(int r=0;r<R;++r) {
        auto gk=local_tile(mk(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
        auto gv=local_tile(mv(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
        tma_copy(p.k,gk,make_tensor(make_smem_ptr(s.k[r].data()),typename C::QL{}),s.kv_full);
        tma_copy(p.v,gv,make_tensor(make_smem_ptr(s.v[r].data()),typename C::QL{}),s.kv_full);
      }
      auto mq=p.q.get_tma_tensor(shape);auto md=p.dout.get_tma_tensor(shape);
      auto mb=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
      for(int jt=0;jt<nt;++jt) {
        int bs=jt%2,bphase=(jt/2)%2;
        s.b_empty[bs].wait(bphase^1);
        s.b_full[bs].arrive_and_expect_tx(4096*sizeof(Element));
        auto gb=local_tile(mb(_,_,h),Shape<_64,_64>{},make_coord(jt,kt));
        tma_copy(p.bias,gb,make_tensor(make_smem_ptr(s.bias[bs].data()),typename C::SL{}),s.b_full[bs]);
        for(int rr=0;rr<C::RP;++rr) {
          int it=jt*C::RP+rr,st=it%2,phase=(it/2)%2;
          for(int w=0;w<2;++w) {
            int row=group*R+w*C::RP+rr;
            s.q_empty[w][st].wait(phase^1);
            s.q_full[w][st].arrive_and_expect_tx(2*2048*sizeof(Element));
            auto gq=local_tile(mq(_,_,h,row),Shape<_64,_32>{},make_coord(jt,0));
            auto gd=local_tile(md(_,_,h,row),Shape<_64,_32>{},make_coord(jt,0));
            tma_copy(p.q,gq,make_tensor(make_smem_ptr(s.q[w][st].data()),typename C::QL{}),s.q_full[w][st]);
            tma_copy(p.dout,gd,make_tensor(make_smem_ptr(s.dout[w][st].data()),typename C::QL{}),s.q_full[w][st]);
          }
        }
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<232>();
  typename C::ScoreMMA smma;typename C::GradMMA gmma;
  auto st=smma.get_slice(lane);auto gt=gmma.get_slice(lane);
  auto gc=partition_fragment_C(gmma,Shape<_64,_32>{});
  auto gcoords=gt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
  auto scoords=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
  constexpr int NG=decltype(size(gc))::value;
  float dkbuf[C::RP][NG],dvbuf[C::RP][NG];
  #pragma unroll
  for(int r=0;r<C::RP;++r) {
    #pragma unroll
    for(int x=0;x<NG;++x) dkbuf[r][x]=dvbuf[r][x]=0;
  }
  auto sp=make_tensor(make_smem_ptr(s.prob[wg].data()),typename C::SL{});
  auto ss=make_tensor(make_smem_ptr(s.ds[wg].data()),typename C::SL{});
  auto tp=st.partition_C(sp);auto ts=st.partition_C(ss);
  s.kv_full.wait(0);
  for(int jt=0;jt<nt;++jt) {
    int bs=jt%2,bphase=(jt/2)%2;
    s.b_full[bs].wait(bphase);
    auto sb=make_tensor(make_smem_ptr(s.bias[bs].data()),typename C::SL{});
    #pragma unroll
    for(int x=0;x<32;++x) s.db[wg][lane*32+x]=0.f;
    // Static row iteration keeps each row's persistent gradient in registers.
    cute::for_each(make_seq<C::RP>{},[&](auto rr_) {
      constexpr int rr=decltype(rr_)::value;
      int r=wg*C::RP+rr,row=group*R+r;
      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;
      s.q_full[wg][slot].wait(phase);
      auto sq=make_tensor(make_smem_ptr(s.q[wg][slot].data()),typename C::QL{});
      auto sd=make_tensor(make_smem_ptr(s.dout[wg][slot].data()),typename C::QL{});
      auto sk=make_tensor(make_smem_ptr(s.k[r].data()),typename C::QL{});
      auto sv=make_tensor(make_smem_ptr(s.v[r].data()),typename C::QL{});
      auto score=partition_fragment_C(smma,Shape<_64,_64>{});
      auto dp=partition_fragment_C(smma,Shape<_64,_64>{});
      auto ka=st.partition_fragment_A(sk);auto qb=st.partition_fragment_B(sq);
      auto va=st.partition_fragment_A(sv);auto dob=st.partition_fragment_B(sd);
      flash::gemm<true,0>(smma,ka,qb,score);
      flash::gemm<true,0>(smma,va,dob,dp);
      #pragma unroll
      for(int x=0;x<size(score);++x) {
        int key=get<0>(scoords(x)),query=get<1>(scoords(x));
        int stat=(h*L+row)*L+jt*64+query;
        float logit=score(x)+float(sb(query,key))/SCALE;
        float pr=exp2f(logit*(SCALE*LOG2E)-fmaxf(p.lse[stat],-1e38f));
        Element ds=Element(pr*(dp(x)-p.delta[stat]));
        tp(x)=Element(pr);ts(x)=ds;
        s.db[wg][lane*32+x]+=float(ds);
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,wg+1);
      auto qt=make_tensor(make_smem_ptr(s.q[wg][slot].data()),typename C::QT{});
      auto dot=make_tensor(make_smem_ptr(s.dout[wg][slot].data()),typename C::QT{});
      auto pa=gt.partition_fragment_A(sp);auto dsa=gt.partition_fragment_A(ss);
      auto qbt=gt.partition_fragment_B(qt);auto dbt=gt.partition_fragment_B(dot);
      auto dk=make_tensor(make_rmem_ptr(dkbuf[rr]),gc.layout());
      auto dv=make_tensor(make_rmem_ptr(dvbuf[rr]),gc.layout());
      flash::gemm<false,0>(gmma,pa,dbt,dv);
      flash::gemm<false,0>(gmma,dsa,qbt,dk);
      cutlass::arch::NamedBarrier::sync(128,wg+1);
      if(lane==0) s.q_empty[wg][slot].arrive();
    });
    cutlass::arch::NamedBarrier::sync(256,3);
    if(wg==0) {
      #pragma unroll
      for(int x=0;x<32;++x) {
        int key=kt*64+get<0>(scoords(x)),query=jt*64+get<1>(scoords(x));
        int64_t idx=((int64_t(h)*(L/R)+group)*L+key)*L+query;
        p.db[idx]=s.db[0][lane*32+x]+s.db[1][lane*32+x];
      }
    }
    cutlass::arch::NamedBarrier::sync(256,3);
    if(lane==0) s.b_empty[bs].arrive();
  }
  #pragma unroll
  for(int rr=0;rr<C::RP;++rr) {
    int row=group*R+wg*C::RP+rr;
    #pragma unroll
    for(int x=0;x<NG;++x) {
      int key=kt*64+get<0>(gcoords(x)),d=get<1>(gcoords(x));
      int idx=(row*L+key)*128+h*32+d;
      p.dk[idx]=Element(dkbuf[rr][x]*SCALE);p.dv[idx]=Element(dvbuf[rr][x]);
    }
  }
}

__global__ void reduce_bias(float const* part,Element* out,int L,int groups) {
  __shared__ float tile[32][33];
  int h=blockIdx.z;
  for(int j=threadIdx.x;j<1024;j+=256) {
    int k=blockIdx.x*32+j/32,q=blockIdx.y*32+j%32;
    float v=0;
    for(int g=0;g<groups;++g) v+=part[((int64_t(h)*groups+g)*L+k)*L+q];
    tile[j/32][j%32]=v;
  }
  __syncthreads();
  for(int j=threadIdx.x;j<1024;j+=256) {
    int q=blockIdx.y*32+j/32,k=blockIdx.x*32+j%32;
    out[(h*L+q)*L+k]=Element(tile[j%32][j/32]);
  }
}

template<int R> std::vector<torch::Tensor> launch(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor delta,torch::Tensor dout) {
  using C=Config<R>;int L=q.size(2);
  auto dk=torch::empty({1,L,L,128},q.options());auto dv=torch::empty_like(dk);
  auto part=torch::empty({4,L/R,L,L},q.options().dtype(torch::kFloat32));
  auto db=torch::empty({1,4,L,L},q.options());
  auto shape=make_shape(L,_32{},_4{},L);
  auto make_q=[&](torch::Tensor const& x) {
    auto mx=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,typename C::StrideQ{x.stride(3),_1{},_32{},x.stride(2)});
    return make_tma_copy(SM90_TMA_LOAD{},mx,typename C::QL{},Shape<_64,_32>{},_1{});
  };
  auto mb=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),typename C::StrideB{L,_1{},int64_t(L)*L});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},mb,typename C::SL{},Shape<_64,_64>{},_1{});
  typename C::Params p{make_q(q),make_q(k),make_q(v),make_q(dout),tb,lse.data_ptr<float>(),delta.data_ptr<float>(),(Element*)dk.data_ptr(),(Element*)dv.data_ptr(),part.data_ptr<float>(),L};
  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv<R>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  auto stream=at::cuda::getCurrentCUDAStream();
  grouped_dkdv<R><<<dim3(L/64,L/R,4),384,sizeof(typename C::Shared),stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_bias<<<dim3(L/32,L/32,4),256,0,stream>>>(part.data_ptr<float>(),(Element*)db.data_ptr(),L,L/R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dk.view({1,L,L,4,32}).permute({0,3,1,2,4}),dv.view({1,L,L,4,32}).permute({0,3,1,2,4}),db};
}

std::vector<torch::Tensor> backward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b,torch::Tensor lse,torch::Tensor delta,torch::Tensor dout,int R) {
  TORCH_CHECK(q.is_cuda() && q.dim()==5 && q.size(0)==1 && q.size(1)==4 && q.size(4)==32,"B1 H4 D32 required");
  c10::cuda::CUDAGuard guard(q.device());int L=q.size(2);
  TORCH_CHECK(L>=64 && L<=1024 && L%64==0 && q.size(3)==L,"square length multiple64, at most1024 required");
  for(auto const& x:{q,k,v,dout}) {
    TORCH_CHECK(x.device()==q.device() && x.scalar_type()==torch::kBFloat16 && x.sizes()==q.sizes(),"QKV/dO shape/device/dtype mismatch");
    TORCH_CHECK(x.stride(4)==1 && x.stride(1)==32 && x.stride(3)==128 && x.stride(2)==L*128,"projection layout required");
  }
  TORCH_CHECK(b.device()==q.device() && b.scalar_type()==torch::kBFloat16 && b.sizes()==torch::IntArrayRef({1,4,L,L}) && b.is_contiguous(),"contiguous BF16 bias required");
  for(auto const& x:{lse,delta}) TORCH_CHECK(x.device()==q.device() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.numel()==4*L*L,"FP32 contiguous statistics required");
  TORCH_CHECK(R==4 || R==8,"row group4 or8 required");
  return R==4?launch<4>(q,k,v,b,lse,delta,dout):launch<8>(q,k,v,b,lse,delta,dout);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);m.def("smem",[](){return std::vector<int>{sizeof(Config<4>::Shared),sizeof(Config<8>::Shared)};});}
