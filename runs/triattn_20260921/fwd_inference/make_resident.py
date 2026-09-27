"""Inference only: resident KV, on-demand Q/gate, one globally stored gated output."""
from pathlib import Path
R=Path(__file__).resolve().parent
base=(R.parent/'fwd_training/qkv_compact_retire6/fused.cu').read_text()
header=base[:base.index('template<int Capacity,int Consumers,int Stages,int Projectors>\n__global__')]
header=header.replace('// Training forward: one score buffer, complete QK groups before scalar score reads.',
                      '// Inference forward: no global Q/K/V/gate/LSE buffers.')
header=header.replace('using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));',
'''using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using WPL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));''')
header=header.replace('ZL{},Shape<_64,_128>{}', 'ZL{},Shape<_64,_64>{}')
header=header.replace('float,Shape<_64,_64,_128>', 'float,Shape<_64,_64,_64>')
header=header.replace('array_aligned<Element,8192,1024> z[Projectors]; array_aligned<Element,8192,1024> w[2];',
                      'array_aligned<Element,4096,1024> z[Consumers]; array_aligned<Element,8192,1024> w;')
header=header.replace('q[Consumers];', 'q[2][Consumers];')
header=header.replace('struct Params { TZ z; TW wq,wg,wk,wv; TB bias; TQ loadq; TO saveq,saveg,savek,savev,out; float* lse; int L; };',
                      'struct Params { TZ z; TW wq,wg,wk,wv; TB bias; TO out; int L; };')
body='''
template<int Capacity,int Consumers,int Stages,int Projectors>
__global__ __launch_bounds__(Consumers*128,1)
void qkv_attention_inference(CUTE_GRID_CONSTANT typename Config<Capacity,Consumers,Stages,Projectors>::Params const p) {
  using C=Config<Capacity,Consumers,Stages,Projectors>;
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x,c=tid/128,lane=tid%128,h=blockIdx.x,row=blockIdx.y,L=p.L,nt=L/64;
  if(tid==0) {
    for(int cc=0;cc<Consumers;++cc)s.zfull[cc].init(1);
    s.wfull.init(1);
    for(int cc=0;cc<Consumers;++cc)for(int st=0;st<Stages;++st)s.bfull[cc][st].init(1);
    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    prefetch_tma_descriptor(p.wq.get_tma_descriptor());prefetch_tma_descriptor(p.wg.get_tma_descriptor());
    prefetch_tma_descriptor(p.wk.get_tma_descriptor());prefetch_tma_descriptor(p.wv.get_tma_descriptor());
    prefetch_tma_descriptor(p.bias.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  auto load_weights=[&](bool kv) {
    if(tid==0) {
      s.wfull.arrive_and_expect_tx(8192*sizeof(Element));
      auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),typename C::WPL{});
      #pragma unroll
      for(int which=0;which<2;++which) {
        auto const& wt=kv?(which==0?p.wk:p.wv):(which==0?p.wq:p.wg);
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
          auto src=local_tile(wg,Shape<_32,_64>{},make_coord(h,chunk));
          auto dst=local_tile(ws,Shape<_32,_64>{},make_coord(which,chunk));
          tma_load(wt,src,dst,s.wfull);
        }
      }
    }
  };
  // All KV remain in shared memory for this head/row. Nothing is saved globally.
  load_weights(true);s.wfull.wait(0);
  {
    typename C::Proj mma;auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z[c].data()),typename C::ZL{});
    auto za=mt.partition_fragment_A(zs);
    auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),typename C::WPL{});
    for(int qt=c;qt<nt;qt+=Consumers) {
      #pragma unroll
      for(int chunk=0;chunk<2;++chunk) {
        if(lane==0) {
          s.zfull[c].arrive_and_expect_tx(4096*sizeof(Element));
          auto zt=local_tile(zg(_,_,row),Shape<_64,_64>{},make_coord(qt,chunk));
          tma_load(p.z,zt,zs,s.zfull[c]);
        }
        s.zfull[c].wait(chunk);
        auto wt=local_tile(ws,Shape<_64,_64>{},make_coord(0,chunk));
        auto wb=mt.partition_fragment_B(wt);
        if(chunk==0)flash::gemm<true,0>(mma,za,wb,acc);
        else flash::gemm<false,0>(mma,za,wb,acc);
        cutlass::arch::NamedBarrier::sync(128,c+1);
      }
      #pragma unroll
      for(int x=0;x<size(acc);x+=2) {
        int mr=get<0>(coord(x)),nd=get<1>(coord(x));
        int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd%32))*2;
        store_pair(cast_smem_ptr_to_uint(s.kv[nd/32][qt].data())+off,acc(x),acc(x+1));
      }
    }
    cutlass::arch::fence_view_async_shared();
  }
  __syncthreads();
  for(int batch=0;batch<nt;batch+=Consumers) {
    int qt=batch+c;
    // Q/gate are produced only for the query tiles about to consume them.
    load_weights(false);s.wfull.wait((1+batch/Consumers)%2);
    {
      typename C::Proj mma;auto mt=mma.get_slice(lane);
      auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
      auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
      if(qt<nt) {
        auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
        auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z[c].data()),typename C::ZL{});
        auto za=mt.partition_fragment_A(zs);
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),typename C::WPL{});
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
          if(lane==0) {
            s.zfull[c].arrive_and_expect_tx(4096*sizeof(Element));
            auto zt=local_tile(zg(_,_,row),Shape<_64,_64>{},make_coord(qt,chunk));
            tma_load(p.z,zt,zs,s.zfull[c]);
          }
          s.zfull[c].wait(chunk);
          auto wt=local_tile(ws,Shape<_64,_64>{},make_coord(0,chunk));
          auto wb=mt.partition_fragment_B(wt);
          if(chunk==0)flash::gemm<true,0>(mma,za,wb,acc);
          else flash::gemm<false,0>(mma,za,wb,acc);
          cutlass::arch::NamedBarrier::sync(128,c+1);
        }
      }
      // Retire every Z/weight reader before compacting Q/gate into the same union.
      __syncthreads();
      if(qt<nt) {
        #pragma unroll
        for(int x=0;x<size(acc);x+=2) {
          int mr=get<0>(coord(x)),nd=get<1>(coord(x));
          int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd%32))*2;
          store_pair(cast_smem_ptr_to_uint(s.scratch.attn.q[nd/32][c].data())+off,acc(x),acc(x+1));
        }
        cutlass::arch::fence_view_async_shared();
      }
    }
    __syncthreads();
    if(qt<nt) {
      auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
      typename C::Score smma;typename C::PV pmma;
      auto st=smma.get_slice(lane);auto pt=pmma.get_slice(lane);
      auto load_bias=[&](int kt) {
        int stage=kt%Stages;
        s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));
        auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
        tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.scratch.attn.bias[c][stage].data()),typename C::BL{}),s.bfull[c][stage]);
      };
      if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);
'''
attention=base[base.index('    auto score='):base.index('    int ep_tid,')]
attention=attention.replace('s.scratch.attn.q[c].data()', 's.scratch.attn.q[0][c].data()')
epi=base[base.index('    int ep_tid,'):base.index('\ntemplate<int Capacity,int Consumers,int Stages,int Projectors>\nvoid launch')]
epi=epi[:epi.rindex('  }\n}')]
begin=epi.index('    auto sc=');end=epi.index('    auto oc=',begin)
epi=epi[:begin]+epi[end:]
begin=epi.index('      if(ep_lane%4==0)');end=epi.index('    uint32_t op=',begin)
epi=epi[:begin]+'    }\n'+epi[end:]
epi=epi.replace('s.scratch.attn.q[ep_c].data()', 's.scratch.attn.q[0][ep_c].data()')
old='''      store_pair(op+off,out(x)*inv[(x%4)/2],out(x+1)*inv[(x%4)/2]);'''
new='''      float r0=float(Element(out(x)*inv[(x%4)/2]));
      float r1=float(Element(out(x+1)*inv[(x%4)/2]));
      float g0=float(s.scratch.attn.q[1][ep_c][off/2]);
      float g1=float(s.scratch.attn.q[1][ep_c][off/2+1]);
      float sg0,sg1,e0=1.f+ex2(-g0*LOG2E),e1=1.f+ex2(-g1*LOG2E);
      asm("rcp.approx.ftz.f32 %0,%1;":"=f"(sg0):"f"(e0));
      asm("rcp.approx.ftz.f32 %0,%1;":"=f"(sg1):"f"(e1));
      store_pair(op+off,r0*sg0,r1*sg1);'''
assert old in epi;epi=epi.replace(old,new)
kernel=header+body+attention+epi+'''    }
    // Output TMA source reads retire before projection overwrites the union.
    __syncthreads();
  }
}
'''
host=base[base.index('template<int Capacity,int Consumers,int Stages,int Projectors>\nvoid launch'):]
host=host.replace('torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor gate,torch::Tensor out,torch::Tensor lse', 'torch::Tensor out')
host=host.replace('typename C::ZL{},Shape<_64,_128>{}', 'typename C::ZL{},Shape<_64,_64>{}')
begin=host.index('  auto qg=');end=host.index('  C10_CUDA_CHECK',begin)
host=host[:begin]+'''  typename C::Params p{tz,weight(wq),weight(wg),weight(wk),weight(wv),tb,save(out),L};
'''+host[end:]
host=host.replace('qkv_attention_compact', 'qkv_attention_inference')
host=host.replace('std::vector<torch::Tensor> launch_forward', 'torch::Tensor launch_forward')
begin=host.index('  auto gate=torch::empty_like(z);');end=host.index('  C10_CUDA_KERNEL_LAUNCH_CHECK();',begin)
host=host[:begin]+'''  auto out=torch::empty_like(z);
  if(L<=128)launch<128,2,2,2>(z,wq,wk,wv,wg,b,out);
  else if(L<=384)launch<384,6,2,6>(z,wq,wk,wv,wg,b,out);
  else if(L==768)launch<768,6,1,6>(z,wq,wk,wv,wg,b,out);
  else launch<1024,6,1,6>(z,wq,wk,wv,wg,b,out);
'''+host[end:]
begin=host.index('  auto view=');end=host.index('\n}',begin)
host=host[:begin]+'  return out;'+host[end:]
host=host.replace('Config<1024,6,1,4>::Shared','Config<1024,6,1,6>::Shared')
for artifact,source in [('resident6',kernel+host),
                        ('resident4',(kernel+host).replace('launch<384,6,2,6>','launch<384,4,2,4>')
                         .replace('launch<768,6,1,6>','launch<768,4,2,4>')
                         .replace('launch<1024,6,1,6>','launch<1024,4,2,4>')
                         .replace('Config<1024,6,1,6>::Shared','Config<1024,4,2,4>::Shared'))]:
    target=R/artifact;assert not (target/'build-ready.json').exists()
    target.mkdir(exist_ok=True);(target/'fused.cu').write_text(source)
