from pathlib import Path
root=Path(__file__).resolve().parent.parent/'dq'
s=(root/'vector_bias/fused.cu').read_text().replace('#include "fa3_utils.h"','#include "fa3_utils.h"\n#include <cute/arch/cluster_sm90.hpp>')
s=s.replace(' using TB=', ' using TK=decltype(make_tma_copy(SM90_TMA_LOAD_MULTICAST{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));\n using TB=')
s=s.replace('ClusterBarrier empty[2];','ClusterBarrier empty[2],kv_empty[2];')
s=s.replace('struct Params {TQ q,k,v,dout;', 'struct Params {TQ q;TK k,v;TQ dout;')
a=s.index('__global__ __launch_bounds__')
s=s[:a]+'''template<class TMA,class G,class S> __device__ __forceinline__ void copy_mc(TMA const& tma,G const& src,S const& dst,cutlass::arch::ClusterTransactionBarrier& ready,uint16_t mask){auto c=tma.get_slice(_0{});copy(tma.with(reinterpret_cast<uint64_t&>(ready),mask),c.partition_S(src),c.partition_D(dst));}
template<int Cluster>
'''+s[a:]
s=s.replace('s.full[i].init(1);s.empty[i].init(1);','s.full[i].init(2);s.empty[i].init(1);s.kv_empty[i].init(Cluster);')
s=s.replace(' __syncthreads();',' __syncthreads();cute::cluster_sync();\n int rank=cute::block_rank_in_cluster();',1)
s=s.replace('s.full[slot].arrive_and_expect_tx((2*2048+4096)*sizeof(Element));','s.full[slot].arrive_and_expect_tx(4096*sizeof(Element));')
old='copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot]);'
new='''if(rank==0){
     s.kv_empty[slot].wait(phase^1);
     #pragma unroll
     for(int target=0;target<Cluster;++target)s.full[slot].arrive_and_expect_tx(2*2048*sizeof(Element),target);
     copy_mc(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot],uint16_t((1<<Cluster)-1));
     copy_mc(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot],uint16_t((1<<Cluster)-1));
    }
    '''
assert old in s;s=s.replace(old,new)
s=s.replace('  return;\n }\n cutlass::arch::warpgroup_reg_alloc<128>();', ' } else {\n cutlass::arch::warpgroup_reg_alloc<128>();')
s=s.replace('if(lane==0)s.empty[slot].arrive();','if(lane==0){s.empty[slot].arrive();s.kv_empty[slot].arrive(0u,1u);}')
s=s.replace('\n}\ntorch::Tensor backward','\n }\n cute::cluster_sync();\n}\ntorch::Tensor backward',1)
a=s.index(' auto bg=make_tensor',s.index('torch::Tensor backward'))
s=s[:a]+''' auto make_k=[&](torch::Tensor const& x){auto src=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,Config::QStride{x.stride(3),_1{},_32{},x.stride(2)});return make_tma_copy(SM90_TMA_LOAD_MULTICAST{},src,Config::QL{},Shape<_64,_32>{},_1{});};
'''+s[a:]
s=s.replace('p{make_q(q),make_q(k),make_q(v),make_q(dy)', 'p{make_q(q),make_k(k),make_k(v),make_q(dy)')
a=s.index(' C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma,');b=s.index('return result.view',a)
s=s[:a]+''' auto launch=[&](auto cl){constexpr int C=decltype(cl)::value;
  C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=C;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(L/64,L,4);cfg.blockDim=dim3(256,1,1);cfg.dynamicSmemBytes=sizeof(Config::Shared);cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=&attr;cfg.numAttrs=1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,dq_tma<C>,p));C10_CUDA_KERNEL_LAUNCH_CHECK();};
 if(L==64)launch(Int<1>{});else launch(Int<2>{});
 '''+s[b:]
d=root/'multicast2';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
