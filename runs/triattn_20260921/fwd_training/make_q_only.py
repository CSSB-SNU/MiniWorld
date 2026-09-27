"""Q projection plus streaming attention; keep the installed attention CTA size."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'cooperative_head2/fused.cu').read_text()
s=s.replace('  using QS=Shape', '''  using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
  using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_128>{}));
  using ZS=Shape<int32_t,_128,int32_t>;using ZStride=Stride<_128,_1,int64_t>;
  using WS=Shape<_128,_128>;using WStride=Stride<_128,_1>;
  using TZ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),ZS{},ZStride{}),ZL{},Shape<_64,_128>{},_1{}));
  using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},WStride{}),WL{},Shape<_32,_128>{},_1{}));
  using Proj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_128>>()));
  using QS=Shape''')
s=s.replace('    array_aligned<Element,2048,1024> q[R], k[2][R], v[2][R];\n    array_aligned<Element,4096,1024> bias[2];', '''    array_aligned<Element,2048,1024> q[R];
    union alignas(1024) {
      struct {array_aligned<Element,8192,1024> z;array_aligned<Element,4096,1024> w;} proj;
      struct {array_aligned<Element,2048,1024> k[2][R],v[2][R];array_aligned<Element,4096,1024> bias[2];} attn;
    } scratch;''')
s=s.replace('struct Params { TQ q,k,v; TB bias; TO out;', 'struct Params { TZ z; TW w; TQ k,v; TB bias; TO saveq,out;')
s=s.replace('training_fwd_stream','qkv_attention_resident_qonly')
s=s.replace('    prefetch_tma_descriptor(p.q.get_tma_descriptor());', '''    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    prefetch_tma_descriptor(p.w.get_tma_descriptor());''')
s=s.replace('  auto qg=p.q.get_tma_tensor(shape), kg=p.k.get_tma_tensor(shape), vg=p.v.get_tma_tensor(shape);', '  auto kg=p.k.get_tma_tensor(shape), vg=p.v.get_tma_tensor(shape);')
for name in ('k','v','bias'):
    s=s.replace('s.'+name+'[','s.scratch.attn.'+name+'[')
old='''  if(tid==0) {
    s.qfull.arrive_and_expect_tx(2048*sizeof(Element));
    auto qq=local_tile(qg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
    tma_load(p.q,qq,make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{}),s.qfull);
    load(0);if(nt>1)load(1);
  }'''
new='''  {
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto wg=p.w.get_tma_tensor(make_shape(_128{},_128{}));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z.data()),Config::ZL{});
    auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),Config::WL{});
    if(tid==0) {
      s.qfull.arrive_and_expect_tx((8192+4096)*sizeof(Element));
      auto zz=local_tile(zg(_,_,rg),Shape<_64,_128>{},make_coord(qt,0));
      auto ww=local_tile(wg,Shape<_32,_128>{},make_coord(h,0));
      tma_load(p.z,zz,zs,s.qfull);tma_load(p.w,ww,ws,s.qfull);
    }
    s.qfull.wait(0);
    Config::Proj mma;auto mt=mma.get_slice(tid);
    auto acc=partition_fragment_C(mma,Shape<_64,_32>{});
    auto za=mt.partition_fragment_A(zs);auto wb=mt.partition_fragment_B(ws);
    flash::gemm<true,0>(mma,za,wb,acc);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
    uint32_t dst=cast_smem_ptr_to_uint(s.q[0].data());
    #pragma unroll
    for(int x=0;x<size(acc);x+=2) {
      int mr=get<0>(coord(x)),nd=get<1>(coord(x));
      int off=as_position_independent_swizzle_layout(Config::QL{})(make_coord(mr,nd))*2;
      store_pair(dst+off,acc(x),acc(x+1));
    }
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::sync(128,0);
    if(tid==0) {
      auto sg=p.saveq.get_tma_tensor(shape);
      auto tile=local_tile(sg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
      auto src=make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{});
      auto cp=p.saveq.get_slice(_0{});
      copy(p.saveq,cp.partition_S(src),cp.partition_D(tile));
      tma_store_arrive();tma_store_wait<0>();
      load(0);if(nt>1)load(1);
    }
  }'''
assert old in s;s=s.replace(old,new)
s=s.replace('    s.qfull.wait(0);\n    auto sq=', '    auto sq=')
start=s.index('std::vector<torch::Tensor> launch_forward(')
end=s.index('  auto shape=make_shape(L,_32{},_4{},L);',start)
s=s[:start]+'''std::vector<torch::Tensor> launch_forward(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor b) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type()==torch::kBFloat16 && z.is_contiguous() && z.dim()==4 && z.size(0)==1 && z.size(3)==128,"contiguous B1 L L C128 BF16 Z required");
  c10::cuda::CUDAGuard guard(z.device());int L=z.size(1);
  TORCH_CHECK(z.size(2)==L && L>=64 && L<=1024 && (L==64 || L%128==0),"unsupported shape");
  for(auto const& w:{wq,wk,wv})TORCH_CHECK(w.device()==z.device() && w.scalar_type()==z.scalar_type() && w.is_contiguous() && w.sizes()==torch::IntArrayRef({128,128}),"invalid projection weight");
  TORCH_CHECK(b.device()==z.device() && b.scalar_type()==z.scalar_type() && b.is_contiguous() && b.sizes()==torch::IntArrayRef({1,4,L,L}),"invalid bias");
  auto q=torch::empty_like(z),k=at::linear(z,wk),v=at::linear(z,wv),output=torch::empty_like(z);
  auto lse=torch::empty({1,4,L,L},z.options().dtype(torch::kFloat32));
  auto zg=make_tensor(make_gmem_ptr((Element const*)z.data_ptr()),make_shape(L,_128{},L),Config::ZStride{_128{},_1{},int64_t(L)*128});
  auto tz=make_tma_copy(SM90_TMA_LOAD{},zg,Config::ZL{},Shape<_64,_128>{},_1{});
  auto wg=make_tensor(make_gmem_ptr((Element const*)wq.data_ptr()),Config::WS{},Config::WStride{});
  auto tw=make_tma_copy(SM90_TMA_LOAD{},wg,Config::WL{},Shape<_32,_128>{},_1{});
''' + s[end:]
s=s.replace('Config::QStride{x.stride(3),_1{},_32{},x.stride(2)}','Config::QStride{128,_1{},_32{},int64_t(L)*128}')
s=s.replace('  Config::Params p{make_q(q),make_q(k),make_q(v),tb,to,lse.data_ptr<float>(),L};', '''  auto qg=make_tensor(make_gmem_ptr((Element*)q.data_ptr()),shape,Config::QStride{128,_1{},_32{},int64_t(L)*128});
  auto saveq=make_tma_copy(SM90_TMA_STORE{},qg,Config::QL{},Shape<_64,_32>{},_1{});
  Config::Params p{tz,tw,make_q(k),make_q(v),tb,saveq,to,lse.data_ptr<float>(),L};''')
s=s.replace('  return {output.view({1,L,L,4,32}).permute({0,3,1,2,4}),lse};', '''  auto view=[&](torch::Tensor const& x){return x.view({1,L,L,4,32}).permute({0,3,1,2,4});};
  return {view(output),lse,view(q),view(k),view(v)};''')
d=root/'q_only_fused';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
