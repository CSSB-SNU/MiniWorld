"""Bounded M64 producer/consumer experiment with a fused N40 PV MMA."""
def transform(s,rows):
 s='#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED\n'+s
 if rows==5:s=s.replace('LN=128','LN=64')
 a=s.index('using SV=');b=s.index('\n',a)
 s=s[:a]+'''using SV=decltype(tile_to_shape(GMMA::Layout_MN_INTER_Atom<Element>{},Shape<_40,Int<LN>>{}));
using SVT=Layout<Shape<_64,_5,Int<LN/8>>,Stride<_1,_64,Int<320>>>;
using GV=Shape<_64,_5,int,int>;
using DV=Stride<_1,_64,Int<320>,int64_t>;
using TV=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GV{},DV{}),SVT{},Shape<_64,_5,Int<LN/8>>{},_1{}));'''+s[b:]
 s=s.replace('SM90_64x32x16_F32BF16BF16_RS','SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS')
 s=s.replace('TQ q;TK k,v;','TQ q;TK k;TV v;')
 s=s.replace('v[Stages][Rows][LN*D]', 'v[Stages][Rows][LN*40]')
 s=s.replace('2*Rows*LN*D*sizeof(Element)', 'Rows*LN*(D+40)*sizeof(Element)')
 s=s.replace('p.v.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0)', 'p.v.get_tma_tensor(make_shape(_64{},_5{},p.L/8,p.L*4))(_,_,_,ih),Shape<_64,_5,Int<LN/8>>{},make_coord(0,0,seq)')
 s=s.replace('auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});', 'auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{});auto vd=make_tensor(make_smem_ptr(s.v[st][r]),SVT{});')
 s=s.replace('auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});','auto kl=p.k.get_slice(_0{});auto vl=p.v.get_slice(_0{});')
 s=s.replace('int wg=tid/128,t=tid%128;', 'int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;')
 s=s.replace('auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);', 'auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);')
 s=s.replace('auto acc=partition_fragment_C(pv,Shape<_64,_32>{});auto sum=partition_fragment_C(ls,Shape<_64,_8>{});', 'auto acc=partition_fragment_C(pv,Shape<_64,_40>{});auto sum=make_tensor(acc.data()+16,decltype(partition_fragment_C(ls,Shape<_64,_8>{})){}.layout());')
 s=s.replace('Shape<_32,Int<N>>{},make_coord(0,seq%Ratio)', 'Shape<_40,Int<N>>{},make_coord(0,seq%Ratio)')
 s=s.replace('  #pragma unroll\n  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),sum);\n','')
 s=s.replace('auto id=tp.partition_C(make_identity_tensor(Shape<_64,_32>{}));', 'auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_40>{}));')
 s=s.replace('for(int n=0;n<size(acc);n+=2)', 'for(int n=0;n<16;n+=2)')
 pos=s.index('__global__ void prepare(')
 s=s[:pos]+'''__global__ void pack_v(Element const* src,Element* dst,int L){
 int64_t n=int64_t(L)*4*L*5;
 for(int64_t ix=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;ix<n;ix+=int64_t(gridDim.x)*blockDim.x){
  int i=ix/(L*5),a=ix%(L*5),kg=a/40,ng=a/8%5,k=kg*8+a%8;
  uint4 value=ng<4?reinterpret_cast<uint4 const*>(src)[(int64_t(i)*L+k)*4+ng]:make_uint4(0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u);
  reinterpret_cast<uint4*>(dst)[ix]=value;
 }
}
'''+s[pos:]
 s=s.replace('auto stream=at::cuda::getCurrentCUDAStream();', '''auto stream=at::cuda::getCurrentCUDAStream();
 auto vp=at::empty({L*4,L,40},q.options());
 pack_v<<<std::min<int64_t>(4096,(int64_t(L)*4*L*5+255)/256),256,0,stream>>>((Element const*)v.data_ptr(),(Element*)vp.data_ptr(),L);''')
 s=s.replace('auto tv=make_tma_copy(SM90_TMA_LOAD{},g(v),SK{},Shape<Int<LN>,_32>{},_1{});', 'auto tv=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)vp.data_ptr()),make_shape(_64{},_5{},L/8,L*4),DV{_1{},_64{},Int<320>{},int64_t(L)*40}),SVT{},Shape<_64,_5,Int<LN/8>>{},_1{});')
 s=s.replace('p.L','768').replace('p.scale','0x1.6a09e6p-3f')
 s=s.replace('using namespace SOL_NAMESPACE;int L=q.size(-2);', 'using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale required");')
 return s
