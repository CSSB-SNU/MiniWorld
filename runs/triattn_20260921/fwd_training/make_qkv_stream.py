"""Stream projected K/V from a dedicated warpgroup to multiple attention consumers."""
import argparse
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--consumers',type=int,choices=(2,4),required=True)
a=ap.parse_args();r=Path(__file__).resolve().parent
old=(r/'qkv_five/fused.cu').read_text()
config=old[:old.index('template<int Capacity,int Consumers,int Stages>\n__global__')]
config=config.replace('Shape<_64,_64>{}));\n  using WL=', 'Shape<_64,_128>{}));\n  using WL=',1)
config=config.replace('ZL{},Shape<_64,_64>{},_1{}', 'ZL{},Shape<_64,_128>{},_1{}')
config=config.replace('Shape<_64,_32,_64>>()));\n  using Score=', 'Shape<_64,_32,_128>>()));\n  using Score=')
b=config.index('  union alignas(1024) Scratch {');e=config.index('\n};',b)
config=config[:b]+'''  union alignas(1024) Scratch {
    struct {array_aligned<Element,8192,1024> z[Consumers];array_aligned<Element,4096,1024> w[2];} proj;
    struct {array_aligned<Element,4096,1024> bias[Consumers][Stages];array_aligned<Element,4096,1024> w[2];} attn;
  };
  struct Shared {
    array_aligned<Element,2048,1024> q[Consumers],kv[2][Stages];
    array_aligned<Element,8192,1024> producer_z;
    Scratch scratch;
    cutlass::arch::ClusterTransactionBarrier zfull[Consumers+1],wfull,bfull[Consumers][Stages];
    cutlass::arch::ClusterBarrier ready[Stages],empty[Stages];
  };
  struct Params {TZ z;TW wq,wk,wv,wg;TB bias;TO saveq,savek,savev,saveg,out;float* lse;int L;};'''+config[e:]
kernel=r'''
template<int Capacity,int Consumers,int Stages>
__global__ __launch_bounds__((Consumers+1)*128,Consumers==2?2:1)
void qkv_attention_stream(CUTE_GRID_CONSTANT typename Config<Capacity,Consumers,Stages>::Params const p) {
  using C=Config<Capacity,Consumers,Stages>;
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x,c=tid/128,lane=tid%128,h=blockIdx.x%4,qgroup=blockIdx.x/4;
  int row=blockIdx.y,L=p.L,nt=L/64;
  if(tid==0) {
    for(int cc=0;cc<=Consumers;++cc)s.zfull[cc].init(1);
    s.wfull.init(1);
    for(int st=0;st<Stages;++st) {
      s.ready[st].init(1);s.empty[st].init(Consumers);
      for(int cc=0;cc<Consumers;++cc)s.bfull[cc][st].init(1);
    }
    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    prefetch_tma_descriptor(p.wq.get_tma_descriptor());prefetch_tma_descriptor(p.wg.get_tma_descriptor());
    prefetch_tma_descriptor(p.wk.get_tma_descriptor());prefetch_tma_descriptor(p.wv.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(c==Consumers)cutlass::arch::warpgroup_reg_dealloc<64>();
  else cutlass::arch::warpgroup_reg_alloc<96>();
  // Each consumer owns one query tile; Q/gate share one normalized-input load.
  if(c<Consumers) {
    int qt=qgroup*Consumers+c;
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z[c].data()),typename C::ZL{});
    if(lane==0) {
      s.zfull[c].arrive_and_expect_tx(8192*sizeof(Element));
      tma_load(p.z,local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(qt,0)),zs,s.zfull[c]);
      if(c==0) {
        s.wfull.arrive_and_expect_tx(2*4096*sizeof(Element));
        for(int which=0;which<2;++which) {
          auto const& wt=which==0?p.wg:p.wq;
          auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
          auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which].data()),typename C::WL{});
          tma_load(wt,local_tile(wg,Shape<_32,_128>{},make_coord(h,0)),ws,s.wfull);
        }
      }
    }
    s.zfull[c].wait(0);s.wfull.wait(0);
    #pragma unroll
    for(int which=0;which<2;++which) {
      typename C::Proj mma;auto mt=mma.get_slice(lane);
      auto acc=partition_fragment_C(mma,Shape<_64,_32>{});
      auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which].data()),typename C::WL{});
      auto za=mt.partition_fragment_A(zs);auto wb=mt.partition_fragment_B(ws);
      flash::gemm<true,0>(mma,za,wb,acc);
      auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
      uint32_t dst=cast_smem_ptr_to_uint(s.q[c].data());
      #pragma unroll
      for(int x=0;x<size(acc);x+=2) {
        int off=as_position_independent_swizzle_layout(typename C::QL{})(coord(x))*2;
        store_pair(dst+off,acc(x),acc(x+1));
      }
      cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,c+1);
      if(lane==0) {
        auto const& save=which==0?p.saveg:p.saveq;
        auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
        auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
        auto src=make_tensor(make_smem_ptr(s.q[c].data()),typename C::QL{});auto cp=save.get_slice(_0{});
        copy(save,cp.partition_S(src),cp.partition_D(tile));tma_store_arrive();tma_store_wait<0>();
      }
      cutlass::arch::NamedBarrier::sync(128,c+1);
    }
  }
  // All Q/gate readers retire before projection scratch becomes bias/weights.
  __syncthreads();
  if(c==Consumers) {
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.producer_z.data()),typename C::ZL{});
    if(lane==0) {
      s.wfull.arrive_and_expect_tx(2*4096*sizeof(Element));
      for(int which=0;which<2;++which) {
        auto const& wt=which==0?p.wk:p.wv;
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        auto ws=make_tensor(make_smem_ptr(s.scratch.attn.w[which].data()),typename C::WL{});
        tma_load(wt,local_tile(wg,Shape<_32,_128>{},make_coord(h,0)),ws,s.wfull);
      }
    }
    s.wfull.wait(1);
    typename C::Proj mma;auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
    auto za=mt.partition_fragment_A(zs);
    auto ws0=make_tensor(make_smem_ptr(s.scratch.attn.w[0].data()),typename C::WL{});
    auto ws1=make_tensor(make_smem_ptr(s.scratch.attn.w[1].data()),typename C::WL{});
    auto bk=mt.partition_fragment_B(ws0),bv=mt.partition_fragment_B(ws1);
    auto ak=partition_fragment_C(mma,Shape<_64,_32>{}),av=partition_fragment_C(mma,Shape<_64,_32>{});
    for(int kt=0;kt<nt;++kt) {
      int st=kt%Stages,phase=(kt/Stages)%2;
      s.empty[st].wait(phase^1);
      if(lane==0) {
        s.zfull[c].arrive_and_expect_tx(8192*sizeof(Element));
        auto ztile=local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(kt,0));
        tma_load(p.z,ztile,zs,s.zfull[c]);
      }
      s.zfull[c].wait(kt%2);
      flash::gemm<true,-1>(mma,za,bk,ak);flash::gemm<true,-1>(mma,za,bv,av);
      warpgroup_wait<0>();warpgroup_fence_operand(ak);warpgroup_fence_operand(av);
      #pragma unroll
      for(int x=0;x<size(ak);x+=2) {
        int off=as_position_independent_swizzle_layout(typename C::QL{})(coord(x))*2;
        store_pair(cast_smem_ptr_to_uint(s.kv[0][st].data())+off,ak(x),ak(x+1));
        store_pair(cast_smem_ptr_to_uint(s.kv[1][st].data())+off,av(x),av(x+1));
      }
      cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,c+1);
      if(lane==0) {
        // Only one query group saves training K/V; every group consumes local tiles.
        if(qgroup==0) {
          #pragma unroll
          for(int which=0;which<2;++which) {
            auto const& save=which==0?p.savek:p.savev;
            auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
            auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
            auto src=make_tensor(make_smem_ptr(s.kv[which][st].data()),typename C::QL{});auto cp=save.get_slice(_0{});
            copy(save,cp.partition_S(src),cp.partition_D(tile));
          }
          tma_store_arrive();tma_store_wait<0>();
        }
        s.ready[st].arrive();
      }
      // Protect producer Z reuse after every issuing thread's WGMMA has retired.
      cutlass::arch::NamedBarrier::sync(128,c+1);
    }
  } else {
'''
attention=old[old.index('  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));',old.index('// QKV saves')):old.index('\ntemplate<int Capacity,int Consumers,int Stages>\nvoid launch')]
attention=attention.replace('s.scratch.bias','s.scratch.attn.bias')
attention=attention.replace('for(int qt=c;qt<nt;qt+=Consumers) {','{\n    int qt=qgroup*Consumers+c;')
start=attention.index('    if(lane==0) {\n      auto qg=')
end=attention.index('    auto score=',start)
attention=attention[:start]+'    if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);\n'+attention[end:]
attention=attention.replace('      auto sk=', '      s.ready[stage].wait((kt/Stages)%2);\n      auto sk=',1)
attention=attention.replace('s.kv[0][kt]','s.kv[0][stage]').replace('s.kv[1][kt]','s.kv[1][stage]')
attention=attention.replace('      if constexpr(Stages==2) {', '''      // QK completion also retires the preceding PV before its slot is released.
      if(lane==0 && kt>0)s.empty[(kt-1)%Stages].arrive();
      if constexpr(Stages==2) {''',1)
attention=attention.replace('((qt/Consumers)*(nt/Stages)+kt/Stages)%2','(kt/Stages)%2')
attention=attention.replace('    float inv[2];','''    cutlass::arch::NamedBarrier::sync(128,c+1);
    if(lane==0)s.empty[(nt-1)%Stages].arrive();
    float inv[2];''')
assert attention.endswith('  }\n}\n')
attention=attention[:-2]+'  }\n}\n'
host=old[old.index('template<int Capacity,int Consumers,int Stages>\nvoid launch'):]
host=host.replace('torch::Tensor wv,torch::Tensor b,','torch::Tensor wv,torch::Tensor wg,torch::Tensor b,')
host=host.replace('torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,','torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor gate,torch::Tensor out,')
host=host.replace('typename C::ZL{},Shape<_64,_64>{}','typename C::ZL{},Shape<_64,_128>{}')
start=host.index('  auto qg=');end=host.index('  C10_CUDA_CHECK',start)
host=host[:start]+'''  typename C::Params p{tz,weight(wq),weight(wk),weight(wv),weight(wg),tb,save(q),save(k),save(v),save(gate),save(out),lse.data_ptr<float>(),L};
'''+host[end:]
host=host.replace('qkv_attention_resident','qkv_attention_stream')
host=host.replace('dim3(4,L),Consumers*128','dim3(4*(L/64/Consumers),L),(Consumers+1)*128')
host=host.replace('auto gate=at::linear(z,wg);','auto gate=torch::empty_like(z);')
b=host.index('  if(L<=128)');e=host.index('  C10_CUDA_KERNEL_LAUNCH_CHECK();',b)
host=host[:b]+'''  if(L==64)launch<64,1,2>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else if(L<=384)launch<384,2,2>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else launch<1024,%d,2>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
'''%a.consumers+host[e:]
host=host.replace('Config<1024,5,1>::Shared','Config<1024,%d,2>::Shared'%a.consumers)
d=r/('qkv_stream_c%d'%a.consumers);assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(config+kernel+attention+host)
