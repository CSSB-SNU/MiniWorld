// Training forward: one score buffer, complete QK groups before scalar score reads.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element = cutlass::bfloat16_t;
constexpr float SCALE=0.1767766952966369f, LOG2E=1.44269504f;
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ void store_pair(uint32_t p,float a,float b) {
  uint32_t v; asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(v):"f"(b),"f"(a));
  asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(v):"memory");
}
struct Config {
  static constexpr int R=1;
  using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
  using VT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using QS=Shape<int32_t,_32,_4,int32_t>;
  using QStride=Stride<int64_t,_1,_32,int64_t>;
  using BS=Shape<int32_t,int32_t,_4>;
  using BStride=Stride<int64_t,_1,int64_t>;
  using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),BS{},BStride{}),BL{},Shape<_64,_64>{},_1{}));
  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));
  using Score=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_32>>()));
  using PV=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  struct Shared {
    array_aligned<Element,2048,1024> q[R], k[3][R], v[3][R];
    array_aligned<Element,4096,1024> bias[3];
    int fast[R];
    cutlass::arch::ClusterTransactionBarrier qfull, full[3][R], bfull[3];
    cutlass::arch::ClusterBarrier empty[3][R], bempty[3];
  };
  struct Params { TQ q,k,v; TB bias; TO out; float* lse; int L; float const* bounds; };
};
template<class T,class G,class S>
__device__ __forceinline__ void tma_load(T const& t,G const& g,S const& s,
    cutlass::arch::ClusterTransactionBarrier& bar) {
  auto c=t.get_slice(_0{});
  copy(t.with(reinterpret_cast<uint64_t&>(bar)),c.partition_S(g),c.partition_D(s));
}

// One K norm maximum per outer row; one live bias maximum per query.
__global__ void training_fwd_prepare_bounds(Element const* k,Element const* bias,float* bounds,int L) {
  int tid=threadIdx.x,lane=tid%32,warp=tid/32,h=blockIdx.y,row=blockIdx.x;
  __shared__ float temp[8];
  if(blockIdx.z==0) {
    float km=0.f;
    for(int key=warp;key<L;key+=8) {
      float x=float(k[(int64_t(row)*L+key)*128+h*32+lane]);
      float n=x*x;
      for(int d=16;d;d/=2)n+=__shfl_xor_sync(0xffffffffu,n,d);
      km=fmaxf(km,n);
    }
    if(lane==0)temp[warp]=km;
    __syncthreads();
    if(tid==0) {
      float mx=0.f;for(int w=0;w<8;++w)mx=fmaxf(mx,temp[w]);
      bounds[h*L+row]=sqrtf(mx)*(1.f+1e-5f)+1e-6f;
    }
  } else {
    float bm=-INFINITY;int count=0;
    for(int key=tid;key<L;key+=256) {
      float b=float(bias[(int64_t(h)*L+row)*L+key]);
      if(b>-1e30f){bm=fmaxf(bm,b);++count;}
    }
    for(int d=16;d;d/=2) {
      bm=fmaxf(bm,__shfl_xor_sync(0xffffffffu,bm,d));
      count+=__shfl_xor_sync(0xffffffffu,count,d);
    }
    __shared__ int counts[8];
    if(lane==0){temp[warp]=bm;counts[warp]=count;}
    __syncthreads();
    if(tid==0) {
      float mx=-INFINITY;int n=0;
      for(int w=0;w<8;++w){mx=fmaxf(mx,temp[w]);n+=counts[w];}
      // Single-key rows retain the original LSE arithmetic and zero adjoints.
      bounds[4*L+h*L+row]=n>1?mx:-INFINITY;
    }
  }
}

template<bool Single> __global__ __launch_bounds__(128,4) void training_fwd_stream(CUTE_GRID_CONSTANT Config::Params const p) {
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<Config::Shared*>(storage);
  int tid=threadIdx.x, qt=blockIdx.x, rg=blockIdx.y, h=blockIdx.z, L=p.L, nt=L/64;
  if(tid==0) {
    s.qfull.init(1);
    for(int r=0;r<Config::R;++r) s.fast[r]=1;
    for(int b=0;b<3;++b) {
      s.bfull[b].init(1); s.bempty[b].init(Config::R);
      for(int r=0;r<Config::R;++r) { s.full[b][r].init(1); s.empty[b][r].init(1); }
    }
    prefetch_tma_descriptor(p.q.get_tma_descriptor());
    prefetch_tma_descriptor(p.k.get_tma_descriptor());
    prefetch_tma_descriptor(p.v.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  auto shape=make_shape(L,_32{},_4{},L);
  auto qg=p.q.get_tma_tensor(shape), kg=p.k.get_tma_tensor(shape), vg=p.v.get_tma_tensor(shape);
  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  auto load=[&](int kt) {
    int stage=kt%3;
    s.full[stage][0].arrive_and_expect_tx(2*2048*sizeof(Element));
    s.bfull[stage].arrive_and_expect_tx(4096*sizeof(Element));
    auto kk=local_tile(kg(_,_,h,rg),Shape<_64,_32>{},make_coord(kt,0));
    auto vv=local_tile(vg(_,_,h,rg),Shape<_64,_32>{},make_coord(kt,0));
    auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
    tma_load(p.k,kk,make_tensor(make_smem_ptr(s.k[stage][0].data()),Config::QL{}),s.full[stage][0]);
    tma_load(p.v,vv,make_tensor(make_smem_ptr(s.v[stage][0].data()),Config::QL{}),s.full[stage][0]);
    tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.bias[stage].data()),Config::BL{}),s.bfull[stage]);
  };
  if(tid==0) {
    s.qfull.arrive_and_expect_tx(2048*sizeof(Element));
    auto qq=local_tile(qg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
    tma_load(p.q,qq,make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{}),s.qfull);
    load(0);if(nt>1)load(1);if(nt>2)load(2);
  }
  {
    int r=0, lane=tid, row=rg;
    Config::Score smma; Config::PV pmma;
    auto st=smma.get_slice(lane); auto pt=pmma.get_slice(lane);
    auto sc=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto oc=pt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
    auto score=partition_fragment_C(smma,Shape<_64,_64>{});
    auto out=partition_fragment_C(pmma,Shape<_64,_32>{}); clear(out);
    float m[2]={-INFINITY,-INFINITY}, l[2]={1.f,1.f};
    s.qfull.wait(0); asm volatile(""::: "memory");
    auto sq=make_tensor(make_smem_ptr(s.q[r].data()),Config::QL{});
    auto qa=st.partition_fragment_A(sq);
    float shifts[2];bool eligible=true;
    float kn=p.bounds[h*L+row];
    uint32_t qptr=cast_smem_ptr_to_uint(s.q[r].data());
    #pragma unroll
    for(int mi=0;mi<2;++mi) {
      int qr=get<0>(sc(2*mi));float qn=0.f;
      #pragma unroll
      for(int d=lane%4;d<32;d+=4) {
        uint32_t addr=qptr+as_position_independent_swizzle_layout(Config::QL{})(make_coord(qr,d))*2;
        uint16_t bits;asm volatile("ld.shared.b16 %0,[%1];":"=h"(bits):"r"(addr):"memory");
        float x=float(Element::bitcast(bits));qn+=x*x;
      }
      qn+=__shfl_xor_sync(0xffffffffu,qn,1);qn+=__shfl_xor_sync(0xffffffffu,qn,2);
      float A=SCALE*sqrtf(qn)*kn;
      float bm=p.bounds[4*L+h*L+qt*64+qr];
      float ub=(bm+A)*LOG2E;
      shifts[mi]=ceilf(ub)+2.f;
      eligible=eligible && isfinite(ub) && (2.f*A*LOG2E<64.f) && fabsf(bm)<1e4f;
    }
    if(!__all_sync(0xffffffffu,eligible) && lane%32==0)atomicExch(&s.fast[r],0);
    cutlass::arch::NamedBarrier::sync(128,r+1);
    bool fast=s.fast[r]!=0;
    if(fast){m[0]=shifts[0];m[1]=shifts[1];l[0]=l[1]=0.f;}

    for(int kt=0;kt<nt;++kt) {
        int stage=kt%3;
        s.full[stage][r].wait((kt/3)%2);
        auto sk=make_tensor(make_smem_ptr(s.k[stage][r].data()),Config::QL{});
        auto kb=st.partition_fragment_B(sk);
        flash::gemm<true,0>(smma,qa,kb,score);
        s.bfull[stage].wait((kt/3)%2);
        asm volatile("":::"memory");
        uint32_t bp=cast_smem_ptr_to_uint(s.bias[stage].data());
        float mx[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int xb=0;xb<size(score);xb+=8) {
          int qr=(lane/32)*16+(lane%16), kr=xb*2+(lane%32/16)*8;
          uint32_t addr=bp+as_position_independent_swizzle_layout(Config::BL{})(make_coord(qr,kr))*2;
          uint32_t bv[4];
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
            :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
          #pragma unroll
          for(int xi=0;xi<8;++xi) {
            int x=xb+xi, mi=(x%4)/2;
            float bias=float(Element::bitcast(uint16_t(bv[xi/2]>>((xi%2)*16))));
            score(x)=score(x)+bias*(1.f/SCALE);
            if(!fast) mx[mi]=fmaxf(mx[mi],score(x));
          }
        }
        float alpha[2], ls[2]={0.f,0.f};
        #pragma unroll
        for(int mi=0;mi<2;++mi) {
          if(fast){alpha[mi]=1.f;continue;}
          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync(0xffffffffu,mx[mi],1));
          if(fast){alpha[mi]=1.f;continue;}
          mx[mi]=fmaxf(mx[mi],__shfl_xor_sync(0xffffffffu,mx[mi],2));
          float mn=fmaxf(fmaxf(m[mi],mx[mi]*(SCALE*LOG2E)),-1e38f);
          alpha[mi]=ex2(m[mi]-mn); m[mi]=mn;
        }
        #pragma unroll
        for(int x=0;x<size(score);++x) {
          int mi=(x%4)/2;
          score(x)=ex2(score(x)*(SCALE*LOG2E)-m[mi]);
          ls[mi]+=score(x);
        }
        #pragma unroll
        for(int mi=0;mi<2;++mi) {
          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],1);
          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],2);
          if(fast)l[mi]+=ls[mi];else l[mi]=l[mi]*alpha[mi]+ls[mi];
        }
        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<Config::PV>(score.layout()));
        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);
        warpgroup_wait<0>(); warpgroup_fence_operand(out);
        #pragma unroll
        for(int x=0;x<size(out);++x) if(!fast)out(x)*=alpha[(x%4)/2];
        // Every reader retires before the producer can overwrite a stage.
        cutlass::arch::NamedBarrier::sync(128,r+1);
        if(tid==0 && kt>0 && kt+2<nt)load(kt+2);
        auto vv=make_tensor(make_smem_ptr(s.v[stage][r].data()),Config::VT{});
        auto vb=pt.partition_fragment_B(vv);
        flash::gemm<false,-1>(pmma,pr,vb,out);
    }
    warpgroup_wait<0>(); warpgroup_fence_operand(out);
    float inv[2];
    #pragma unroll
    for(int mi=0;mi<2;++mi) {
      float den=l[mi]>0.f?l[mi]:1.f; inv[mi]=1.f/den;
      if(lane%4==0) {
        int qr=get<0>(sc(2*mi));
        p.lse[(h*L+row)*L+qt*64+qr]=m[mi]+log2f(den);
      }
    }
    uint32_t op=cast_smem_ptr_to_uint(s.q[r].data());
    #pragma unroll
    for(int x=0;x<size(out);x+=2) {
      int qr=get<0>(oc(x)), d=get<1>(oc(x));
      int off=as_position_independent_swizzle_layout(Config::QL{})(make_coord(qr,d))*2;
      store_pair(op+off,out(x)*inv[(x%4)/2],out(x+1)*inv[(x%4)/2]);
    }
    cutlass::arch::fence_view_async_shared(); cutlass::arch::NamedBarrier::sync(128,r+1);
    if(lane==0) {
      auto og=p.out.get_tma_tensor(make_shape(L,_32{},_4{},L));
      auto tile=local_tile(og(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
      auto src=make_tensor(make_smem_ptr(s.q[r].data()),Config::QL{}); auto c=p.out.get_slice(_0{});
      copy(p.out,c.partition_S(src),c.partition_D(tile)); tma_store_arrive(); tma_store_wait<0>();
    }
  }
}
std::vector<torch::Tensor> launch_forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor b) {
  TORCH_CHECK(q.is_cuda() && q.dim()==5 && q.size(0)==1 && q.size(1)==4 && q.size(4)==32,"B1 H4 D32 required");
  c10::cuda::CUDAGuard guard(q.device()); int L=q.size(2);
  TORCH_CHECK(L>=64 && L<=1024 && (L==64 || L%128==0) && q.size(3)==L,"square L multiple64 <=1024");
  for(auto const& t:{q,k,v}) {
    TORCH_CHECK(t.device()==q.device() && t.scalar_type()==torch::kBFloat16 && t.sizes()==q.sizes(),"QKV metadata mismatch");
    TORCH_CHECK(t.stride(4)==1 && t.stride(1)==32 && t.stride(3)==128 && t.stride(2)==L*128,"projection layout required");
  }
  TORCH_CHECK(b.device()==q.device() && b.scalar_type()==torch::kBFloat16 && b.is_contiguous() && b.sizes()==torch::IntArrayRef({1,4,L,L}),"bias metadata mismatch");
  auto output=torch::empty({1,L,L,128},q.options());
  auto lse=torch::empty({1,4,L,L},q.options().dtype(torch::kFloat32));
  auto shape=make_shape(L,_32{},_4{},L);
  auto make_q=[&](torch::Tensor const& x) {
    auto g=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,Config::QStride{x.stride(3),_1{},_32{},x.stride(2)});
    return make_tma_copy(SM90_TMA_LOAD{},g,Config::QL{},Shape<_64,_32>{},_1{});
  };
  auto bg=make_tensor(make_gmem_ptr((Element const*)b.data_ptr()),make_shape(L,L,_4{}),Config::BStride{L,_1{},int64_t(L)*L});
  auto tb=make_tma_copy(SM90_TMA_LOAD{},bg,Config::BL{},Shape<_64,_64>{},_1{});
  auto og=make_tensor(make_gmem_ptr((Element*)output.data_ptr()),shape,Config::QStride{128,_1{},_32{},int64_t(L)*128});
  auto to=make_tma_copy(SM90_TMA_STORE{},og,Config::QL{},Shape<_64,_32>{},_1{});
  auto bounds=torch::empty({2,4,L},q.options().dtype(torch::kFloat32));
  training_fwd_prepare_bounds<<<dim3(L,4,2),256,0,at::cuda::getCurrentCUDAStream()>>>((Element const*)k.data_ptr(),(Element const*)b.data_ptr(),bounds.data_ptr<float>(),L);
  Config::Params p{make_q(q),make_q(k),make_q(v),tb,to,lse.data_ptr<float>(),L,bounds.data_ptr<float>()};
  if(L==64) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(training_fwd_stream<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));
    training_fwd_stream<true><<<dim3(L/64,L/Config::R,4),128,sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(training_fwd_stream<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));
    training_fwd_stream<false><<<dim3(L/64,L/Config::R,4),128,sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output.view({1,L,L,4,32}).permute({0,3,1,2,4}),lse};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&launch_forward); m.def("smem",[](){return sizeof(Config::Shared);});
}
