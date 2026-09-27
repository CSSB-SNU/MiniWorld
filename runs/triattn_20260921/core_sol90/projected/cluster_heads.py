"""Four head CTAs share normalized X via full-tile TMA multicast."""
def transform(s):
    s=s.replace('#include <cute/tensor.hpp>','#include <cute/tensor.hpp>\n#include <cute/arch/cluster_sm90.hpp>\n#include <cutlass/cluster_launch.hpp>')
    # Logical TMA tiler still owns the full X tile; multicast replicates it.
    s=s.replace('using TX=decltype(make_tma_copy(SM90_TMA_LOAD{},','using TX=decltype(make_tma_copy(SM90_TMA_LOAD_MULTICAST{},')
    s=s.replace('auto tx=make_tma_copy(SM90_TMA_LOAD{},','auto tx=make_tma_copy(SM90_TMA_LOAD_MULTICAST{},')
    s=s.replace('ClusterBarrier qr,full[Stages],empty[Stages],be[8];','ClusterBarrier qr,full[Stages],empty[Stages],be[8],xfree[Rows];')
    s=s.replace('for(int r=0;r<Rows;++r)s.xr[r].init(1);','for(int r=0;r<Rows;++r){s.xr[r].init(1);s.xfree[r].init(4);}')
    old='int tid=threadIdx.x,qt=tile%6,rg=(tile/6)%(L/Rows),h=tile/(6*(L/Rows)),i0=rg*Rows;'
    assert old in s
    s=s.replace(old,'int tid=threadIdx.x,h=tile%4,qt=(tile/4)%6,rg=tile/(4*6),i0=rg*Rows;')
    s=s.replace('if(!p.fix[tile])return;','if(!p.fix[tile/4])return;')
    s=s.replace('atomicExch(p.fix+tile,1)','atomicExch(p.fix+tile/4,1)')
    mark='cutlass::arch::fence_view_async_shared();__syncthreads();\n'
    assert s.count(mark)==1
    s=s.replace(mark,mark+' cute::cluster_sync();\n')
    s=s.replace('auto load_x=[&](int r,int kt)','auto load_x=[&](int r,int kt,int phase)')
    old='''  s.xr[r].arrive_and_expect_tx(128*128*sizeof(Element));
  copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr[r])),slice.partition_S(src),slice.partition_D(dst));'''
    assert old in s
    s=s.replace(old,'''  // Every CTA arms its local transaction barrier before telling rank0 it
  // has finished reading the old X contents. Only rank0 issues the multicast.
  s.xr[r].arrive_and_expect_tx(128*128*sizeof(Element));
  s.xfree[r].arrive(uint32_t(0),uint32_t(1));
  if(h==0){
   s.xfree[r].wait(phase);asm volatile("":::"memory");
   copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr[r]),uint16_t(15)),slice.partition_S(src),slice.partition_D(dst));
  }''')
    s=s.replace('load_x(0,qt);load_x(1,qt);','load_x(0,qt,0);load_x(1,qt,0);')
    s=s.replace('load_x(0,0);load_x(1,0);','load_x(0,0,1);load_x(1,0,1);')
    s=s.replace('load_x(r,kt+1);','load_x(r,kt+1,(kt+2)&1);')
    old=' return;\n}\n cutlass::arch::warpgroup_reg_alloc'
    assert old in s
    s=s.replace(old,'}\nelse{\n cutlass::arch::warpgroup_reg_alloc')
    marker='__global__ void prepare('
    pos=s.index(marker)
    body=s[:pos].rstrip()
    assert body.endswith('}')
    # All producer and consumer lanes remain alive until the whole cluster
    # is finished, preserving the lifetime of remote shared barriers.
    body=body[:-1]+'}\n cute::cluster_sync();\n}\n'
    s=body+s[pos:]
    old=''' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);'''
    assert old in s
    s=s.replace(old,''' cutlass::ClusterLaunchParams launch{dim3(ct),dim3(Threads),dim3(4,1,1),int(sizeof(Shared)),stream};
 TORCH_CHECK(cutlass::launch_kernel_on_cluster(launch,(void const*)attention<false>,p)==cutlass::Status::kSuccess,"hot cluster launch failed");
 TORCH_CHECK(cutlass::launch_kernel_on_cluster(launch,(void const*)attention<true>,p)==cutlass::Status::kSuccess,"SAFE cluster launch failed");''')
    return s
