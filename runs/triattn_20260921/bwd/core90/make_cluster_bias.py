"""Test two R4 CTAs sharing dS through DSM to retain R8 global partial traffic.

Each CTA keeps its four final dK/dV outputs. Rank0 combines eight rounded dS
contributions in FP32. Cluster readiness and read-completion barriers protect
both double-buffered dS slots; all CTAs stay alive through the final DSM access.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
s = (root / 'bias_fusion/rs_tma_store/grouped.cu').read_text()
s = s.replace('#include <cute/tensor.hpp>',
              '#include <cute/tensor.hpp>\n#include <cute/arch/cluster_sm90.hpp>')
s = s.replace('cutlass::arch::ClusterBarrier q_empty[NWG][2],b_empty[2];',
              'cutlass::arch::ClusterBarrier q_empty[NWG][2],b_empty[2],ds_ready[2],ds_empty[2];')
s = s.replace('s.b_full[st].init(1);s.b_empty[st].init(C::NWG);',
              's.b_full[st].init(1);s.b_empty[st].init(C::NWG);s.ds_ready[st].init(2*C::NWG);s.ds_empty[st].init(2*C::NWG);')
s = s.replace('  __syncthreads();\n  if(tid<128)',
              '  __syncthreads();cute::cluster_sync();\n  int rank=cute::block_rank_in_cluster();\n  if(tid<128)')
s = s.replace('    return;\n  }\n  cutlass::arch::warpgroup_reg_alloc<112>();',
              '  } else {\n  cutlass::arch::warpgroup_reg_alloc<112>();')
old = '''    cutlass::arch::NamedBarrier::sync(512,5);
    { // All four consumers reduce one quarter of the bias tile each.'''
new = '''    cutlass::arch::NamedBarrier::sync(512,5);
    // Each warpgroup publishes ordinary shared writes to both CTA barriers.
    if(lane==0){s.ds_ready[bs].arrive(0u,1u);s.ds_ready[bs].arrive(1u,1u);}
    s.ds_ready[bs].wait(bphase);
    if(rank==0) { // Rank0 writes one R8 partial for the paired R4 CTAs.'''
assert old in s
s = s.replace(old, new)
old = '''        #pragma unroll
        for(int w=0;w<C::NWG;++w) {
          uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs].data())+off;
          asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
          sum0+=float(Element::bitcast(uint16_t(pair)));
          sum1+=float(Element::bitcast(uint16_t(pair>>16)));
        }'''
new = '''        #pragma unroll
        for(int peer=0;peer<2;++peer){
          #pragma unroll
          for(int w=0;w<C::NWG;++w) {
            uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs].data())+off;
            if(peer==0){asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");}
            else {ptr=cute::set_block_rank(ptr,1u);asm volatile("ld.shared::cluster.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");}
            sum0+=float(Element::bitcast(uint16_t(pair)));
            sum1+=float(Element::bitcast(uint16_t(pair>>16)));
          }
        }'''
assert old in s
s = s.replace(old, new)
s = s.replace('int64_t(h)*(L/R)+group', 'int64_t(h)*(L/(2*R))+group/2')
old = '''    // The other dS slot protects readers until the next pre-reduction barrier.
    if(lane==0) s.b_empty[bs].arrive();'''
new = '''    // Rank1 must not recycle dS before every rank0 DSM reader finishes.
    cutlass::arch::NamedBarrier::sync(512,5);
    if(lane==0){s.ds_empty[bs].arrive(0u,1u);s.ds_empty[bs].arrive(1u,1u);}
    s.ds_empty[bs].wait(bphase);
    if(lane==0) s.b_empty[bs].arrive();'''
assert old in s
s = s.replace(old, new)
old = '\n}\n\n__global__ void reduce_bias'
new = '\n  } // consumer branch\n  cute::cluster_sync(); // Keep both DSM allocations alive through the final read.\n}\n\n__global__ void reduce_bias'
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('torch::empty({4,L/R,L,L}', 'torch::empty({4,L/(2*R),L,L}')
old = '  grouped_dkdv<R><<<dim3(L/64,L/R,4),640,sizeof(typename C::Shared),stream>>>(p);'
new = '''  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x=1;attr.val.clusterDim.y=2;attr.val.clusterDim.z=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(L/64,L/R,4);cfg.blockDim=dim3(640,1,1);
  cfg.dynamicSmemBytes=sizeof(typename C::Shared);cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,grouped_dkdv<R>,p));'''
assert old in s
s = s.replace(old, new)
s = s.replace('(Element*)db.data_ptr(),L,L/R);', '(Element*)db.data_ptr(),L,L/(2*R));')
s = s.replace('TORCH_CHECK(R==4,"row group4 required");',
              'TORCH_CHECK(R==4 || R==8,"row group4 legacy or actual8 required");')
s = s.replace('m.def("backward",&backward);', 'm.attr("row_group")=8;m.def("backward",&backward);')
s = s.replace('// One TMA producer, four consumer warpgroups; each WG owns one outer row.',
              '// Cluster of two R4 CTAs: one TMA producer and four consumer warpgroups per CTA.')
target = root / 'bias_fusion/rs4_cluster2'
target.mkdir(exist_ok=True)
(target / 'grouped.cu').write_text(s)
