"""Share streamed K/V across adjacent query tiles of the SAME outer row.

Unlike the rejected grouped-row variants, all consumer WGs read the same K/V.
Each owns its final dQ and resident Q/dO/stats. No global partial is introduced.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'dq'
base = (root / 'rs_softmax_overlap/fused.cu').read_text()
consumer = base[base.index(' int lane=tid;Config::Score'):base.index('\n}\ntorch::Tensor backward')]
consumer = consumer.replace(' int lane=tid;Config::Score', ' Config::Score')
for name in ('q', 'dout', 'lse', 'delta'):
    consumer = consumer.replace('s.%s.data()' % name, 's.%s[wg].data()' % name)
consumer = consumer.replace('s.bias[slot].data()', 's.bias[wg][slot].data()')
consumer = consumer.replace('  if(tid==0 && kt+1<L/64)load(kt+1);\n', '')
consumer = consumer.replace('flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncthreads();',
                            'flash::gemm<false,0>(gmma,dsa,ktb,dq);cutlass::arch::NamedBarrier::sync(128,wg+1);if(lane==0)s.empty[slot].arrive();')
consumer = consumer.replace('cutlass::arch::fence_view_async_shared();__syncthreads();',
                            'cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);')
assert '__syncthreads' not in consumer and 'load(kt+' not in consumer

grouped = '''
template<int G> struct QueryShared {
 array_aligned<Element,2048,1024> q[G],dout[G],k[2],v[2];
 array_aligned<Element,4096,1024> bias[G][2];
 array_aligned<float,64,128> lse[G],delta[G];
 cutlass::arch::ClusterTransactionBarrier resident,full[2];
 cutlass::arch::ClusterBarrier empty[2];
};
template<int G> __global__ __launch_bounds__((G+1)*128,(G==2?2:1))
void dq_tma_grouped(CUTE_GRID_CONSTANT Config::Params const p){
 extern __shared__ char storage[];auto& s=*reinterpret_cast<QueryShared<G>*>(storage);
 int tid=threadIdx.x,lane=tid%128,wg=tid/128-1,row=blockIdx.y,h=blockIdx.z;
 int qbase=blockIdx.x*G,qt=qbase+wg,L=p.L;
 if(tid==0){
  s.resident.init(1);
  for(int st=0;st<2;++st){s.full[st].init(1);s.empty[st].init(G);}
  prefetch_tma_descriptor(p.q.get_tma_descriptor());prefetch_tma_descriptor(p.dout.get_tma_descriptor());
  prefetch_tma_descriptor(p.k.get_tma_descriptor());prefetch_tma_descriptor(p.v.get_tma_descriptor());
  prefetch_tma_descriptor(p.bias.get_tma_descriptor());cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if(tid<128){
  cutlass::arch::warpgroup_reg_dealloc<32>();
  if(tid==0){
   auto shape=make_shape(L,_32{},_4{},L);
   auto qg=p.q.get_tma_tensor(shape);auto dog=p.dout.get_tma_tensor(shape);
   s.resident.arrive_and_expect_tx(G*(2*2048*sizeof(Element)+2*64*sizeof(float)));
   for(int w=0;w<G;++w){
    auto qtile=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qbase+w,0));
    auto dotile=local_tile(dog(_,_,h,row),Shape<_64,_32>{},make_coord(qbase+w,0));
    copy_tile(p.q,qtile,make_tensor(make_smem_ptr(s.q[w].data()),Config::QL{}),s.resident);
    copy_tile(p.dout,dotile,make_tensor(make_smem_ptr(s.dout[w].data()),Config::QL{}),s.resident);
    int stat=(h*L+row)*L+(qbase+w)*64;
    SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.resident),s.lse[w].data(),64*sizeof(float));
    SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.resident),s.delta[w].data(),64*sizeof(float));
   }
   auto kg=p.k.get_tma_tensor(shape);auto vg=p.v.get_tma_tensor(shape);
   auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
   for(int kt=0;kt<L/64;++kt){
    int slot=kt%2,phase=(kt/2)%2;
    s.empty[slot].wait(phase^1);
    s.full[slot].arrive_and_expect_tx((2*2048+G*4096)*sizeof(Element));
    auto kk=local_tile(kg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
    auto vv=local_tile(vg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
    copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot]);
    copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot]);
    for(int w=0;w<G;++w){
     auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qbase+w,kt));
     copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[w][slot].data()),Config::SL{}),s.full[slot]);
    }
   }
  }
  return;
 }
 cutlass::arch::warpgroup_reg_alloc<(G==2?104:112)>();
''' + consumer + '''
}
template<int G> void launch_query_group(Config::Params const& p){
 C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma_grouped<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(QueryShared<G>)));
 dq_tma_grouped<G><<<dim3(p.L/(64*G),p.L,4),(G+1)*128,sizeof(QueryShared<G>),at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

base = base.replace('dq_tma', 'dq_tma_single')
base = base.replace('torch::Tensor backward(', grouped + '\ntorch::Tensor backward(', 1)
start = base.index(' C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma_single,')
end = base.index('return result.view', start)
single_launch = base[start:end]
for g in (2, 4):
    if g == 2:
        launch = ' if(L%128==0){launch_query_group<2>(p);}else{' + single_launch + '}\n '
    else:
        launch = ' if(L%256==0){launch_query_group<4>(p);}else if(L%128==0){launch_query_group<2>(p);}else{' + single_launch + '}\n '
    source = base[:start] + launch + base[end:]
    source = source.replace('// Query-owned dQ; resident Q/dO/stats, double-buffered TMA K/V/bias.',
                            '// Adjacent query tiles share K/V; each WG owns final dQ, resident Q/dO/stats.')
    source = source.replace('return sizeof(Config::Shared);',
                            'return std::vector<int>{sizeof(Config::Shared),sizeof(QueryShared<2>),sizeof(QueryShared<4>)};')
    target = root / ('rs_query_group%d' % g)
    target.mkdir(exist_ok=True)
    (target / 'fused.cu').write_text(source)
