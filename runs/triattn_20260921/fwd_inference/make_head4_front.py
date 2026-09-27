"""Fuse LN/bias and all-head KV, retaining the tested tiled QG/attention core."""
from pathlib import Path
import argparse
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--artifact',default='h4kv_front1')
ap.add_argument('--vectorized',action='store_true')
a=ap.parse_args()
s=(R/'h4kv_local1/fused.cu').read_text()
f=(R/'head4_kv_projection.cuh').read_text()
f=f[:f.index('void project_head4_kv(')]
f=f.replace('Head4KV','FrontKV').replace('head4_kv_projection','front_head4_kv_kernel')
f=f.replace('  struct Params { TZ z; TW wk,wv; TO key,value; int rows; };', '''  using XS=Shape<int32_t,_128,int32_t>;using XStride=Stride<int64_t,_1,int64_t>;
  using TX=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),XS{},XStride{}),ZL{},Shape<_64,_128>{},_1{}));
  struct Params { TX x; TW wk,wv; TO key,value,norm; void const* gamma;void const* beta;
    Element const* wb;bool const* mask;Element* bias;float eps;int L,rows; };''')
f=f.replace('__global__ __launch_bounds__(128,2)', 'template<class G>\n__global__ __launch_bounds__(128,2)')
old='''        auto g=p.z.get_tma_tensor(make_shape(p.rows,_128{}));
        auto tile=local_tile(g,Shape<_64,_128>{},make_coord(blockIdx.x,0));
        tma_load(p.z,tile,zs,s.full);'''
new='''        auto g=p.x.get_tma_tensor(make_shape(p.L,_128{},p.L));
        auto tile=local_tile(g(_,_,int(blockIdx.x)/(p.L/64)),Shape<_64,_128>{},make_coord(int(blockIdx.x)%(p.L/64),0));
        tma_load(p.x,tile,zs,s.full);'''
assert old in f;f=f.replace(old,new)
f=f.replace('    s.full.wait(which);', '''    s.full.wait(which);
    if(which==0) {
      int lc=lane%8;
      auto gamma=static_cast<G const*>(p.gamma),beta=static_cast<G const*>(p.beta);
      for(int rr=lane/8;rr<64;rr+=16) {
        float values[16],mu=0.f;
        #pragma unroll
        for(int t=0;t<16;++t) {values[t]=float(zs(rr,lc*16+t));mu+=values[t];}
        #pragma unroll
        for(int delta=4;delta;delta/=2)mu+=__shfl_xor_sync(0xffffffffu,mu,delta);
        mu*=1.f/128;float var=0.f;
        #pragma unroll
        for(int t=0;t<16;++t){values[t]-=mu;var+=values[t]*values[t];}
        #pragma unroll
        for(int delta=4;delta;delta/=2)var+=__shfl_xor_sync(0xffffffffu,var,delta);
        float inv=rsqrtf(var*(1.f/128)+p.eps);
        #pragma unroll
        for(int t=0;t<16;++t){
          float norm=values[t]*inv;
          values[t]=float(Element(fmaf(norm,float(gamma[lc*16+t]),float(beta[lc*16+t]))));
          zs(rr,lc*16+t)=Element(values[t]);
        }
        #pragma unroll
        for(int h=0;h<4;++h){
          float bias=0.f;
          #pragma unroll
          for(int t=0;t<16;++t)bias=fmaf(values[t],float(p.wb[h*128+lc*16+t]),bias);
          #pragma unroll
          for(int delta=4;delta;delta/=2)bias+=__shfl_xor_sync(0xffffffffu,bias,delta);
          int r=int(blockIdx.x)*64+rr;
          if(lc==0)p.bias[h*p.rows+r]=Element(p.mask && !p.mask[r%p.L]?-3.3895313892515355e38f:bias);
        }
      }
      cutlass::arch::fence_view_async_shared();__syncthreads();
      if(lane==0){
        auto g=p.norm.get_tma_tensor(make_shape(p.rows,_128{}));
        auto tile=local_tile(g,Shape<_64,_128>{},make_coord(blockIdx.x,0));auto ts=p.norm.get_slice(_0{});
        copy(p.norm,ts.partition_S(zs),ts.partition_D(tile));tma_store_arrive();
      }
    }''')
f+='''
std::vector<torch::Tensor> front_head4_kv(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,
    torch::Tensor wb,torch::Tensor mask,double eps,bool ending,torch::Tensor wk,torch::Tensor wv) {
  using C=FrontKV;
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.is_contiguous() &&
    x.dim()==4 && x.size(0)==1 && x.size(1)==x.size(2) && x.size(3)==128,"invalid input");
  c10::cuda::CUDAGuard guard(x.device());int L=x.size(1),rows=L*L;
  TORCH_CHECK(L%64==0,"L must be a multiple of 64");
  for(auto w:{gamma,beta,wb,wk,wv})TORCH_CHECK(w.device()==x.device() && w.is_contiguous(),"invalid weight");
  TORCH_CHECK(gamma.numel()==128 && beta.numel()==128 && gamma.scalar_type()==beta.scalar_type() &&
    (gamma.scalar_type()==torch::kFloat32 || gamma.scalar_type()==torch::kBFloat16),"invalid affine");
  for(auto w:{wk,wv})TORCH_CHECK(w.scalar_type()==x.scalar_type() && w.sizes()==torch::IntArrayRef({128,128}),"invalid KV weight");
  TORCH_CHECK(wb.scalar_type()==x.scalar_type() && wb.sizes()==torch::IntArrayRef({4,128}),"invalid bias weight");
  TORCH_CHECK(mask.device()==x.device() && mask.scalar_type()==torch::kBool && mask.is_contiguous() &&
    (mask.numel()==0 || mask.sizes()==torch::IntArrayRef({1,L})),"invalid mask");
  auto z=torch::empty_like(x),key=torch::empty_like(x),value=torch::empty_like(x);
  auto bias=torch::empty({1,4,L,L},x.options());
  auto xg=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,_128{},L),
    C::XStride{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128});
  auto tx=make_tma_copy(SM90_TMA_LOAD{},xg,C::ZL{},Shape<_64,_128>{},_1{});
  auto weight=[&](torch::Tensor const& w) {
    auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),C::WS{},C::Stride2{});
    return make_tma_copy(SM90_TMA_LOAD{},g,C::WL{},Shape<_128,_128>{},_1{});
  };
  auto output=[&](torch::Tensor const& o) {
    auto g=make_tensor(make_gmem_ptr((Element*)o.data_ptr()),make_shape(rows,_128{}),C::Stride2{});
    return make_tma_copy(SM90_TMA_STORE{},g,C::OL{},Shape<_64,_128>{},_1{});
  };
  C::Params p{tx,weight(wk),weight(wv),output(key),output(value),output(z),gamma.data_ptr(),beta.data_ptr(),
    (Element const*)wb.data_ptr(),mask.numel()?mask.data_ptr<bool>():nullptr,(Element*)bias.data_ptr(),float(eps),L,rows};
  auto run=[&](auto tag){
    using G=decltype(tag);
    C10_CUDA_CHECK(cudaFuncSetAttribute(front_head4_kv_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(C::Shared)));
    front_head4_kv_kernel<G><<<rows/64,128,sizeof(C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  };
  if(gamma.scalar_type()==torch::kFloat32)run(float{});else run(Element{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {z,bias,key,value};
}
'''
if a.vectorized:
    f=f.replace('''        float values[16],mu=0.f;
        #pragma unroll
        for(int t=0;t<16;++t) {values[t]=float(zs(rr,lc*16+t));mu+=values[t];}''', '''        uint4 packed_input[2],packed_norm[2];
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk){
          int offset=as_position_independent_swizzle_layout(C::ZL{})(make_coord(rr,lc*16+chunk*8));
          packed_input[chunk]=*reinterpret_cast<uint4 const*>(s.z.data()+offset);
        }
        auto input=reinterpret_cast<Element*>(packed_input),norm_out=reinterpret_cast<Element*>(packed_norm);
        float values[16],mu=0.f;
        #pragma unroll
        for(int t=0;t<16;++t) {values[t]=float(input[t]);mu+=values[t];}''')
    f=f.replace('''          zs(rr,lc*16+t)=Element(values[t]);
        }
        #pragma unroll
        for(int h=0;h<4;++h){''', '''          norm_out[t]=Element(values[t]);
        }
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk){
          int offset=as_position_independent_swizzle_layout(C::ZL{})(make_coord(rr,lc*16+chunk*8));
          *reinterpret_cast<uint4*>(s.z.data()+offset)=packed_norm[chunk];
        }
        #pragma unroll 1
        for(int h=0;h<4;++h){''')
pos=s.index('template<int Capacity,int Consumers,int Stages,int Projectors>\n__global__')
s=s[:pos]+f+'\n'+s[pos:]
s=s.replace('torch::Tensor counts={}) {', 'torch::Tensor counts={},torch::Tensor key={},torch::Tensor value={}) {')
s=s.replace('''  auto key=torch::empty_like(z),value=torch::empty_like(z);
  project_head4_kv(z,wk,wv,key,value);''', '''  if(!key.defined()) {
    key=torch::empty_like(z);value=torch::empty_like(z);
    project_head4_kv(z,wk,wv,key,value);
  }''')
s=s.replace('  m.def("forward_audit",&launch_forward);', '''  m.def("forward_audit",[](torch::Tensor z,torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor b,torch::Tensor counts){return launch_forward(z,q,k,v,g,b,counts);});
  m.def("front_kv",&front_head4_kv);
  m.def("forward_kv",[](torch::Tensor z,torch::Tensor q,torch::Tensor g,torch::Tensor b,torch::Tensor key,torch::Tensor value){return launch_forward(z,q,q,q,g,b,{},key,value);});''')
folder=R/a.artifact
assert not(folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)
